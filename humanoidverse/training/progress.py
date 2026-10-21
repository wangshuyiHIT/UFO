"""Atomic numeric training receipts for an independent process supervisor.

File activity is not progress: supervisors must compare the completed counters
and bound time in initialization/evaluation/checkpoint phases separately.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import tempfile
import time
import uuid
import re
from . import metadata_output


SCHEMA = "ufo_training_progress_v1"
PHASES = {"initializing", "warmup", "training", "evaluation", "checkpoint", "finished", "failed"}


def _counter(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _validate_previous(value):
    base = {"schema", "run_id", "pid", "rank", "run_started_unix_s", "observed_at_unix_s", "elapsed_s", "phase", "phase_started_unix_s", "phase_elapsed_s", "global_time", "optimizer_steps", "sequence", "resume_global_time", "resume_optimizer_steps"}
    prior = {"prior_run_id", "resume_rollback_global_time", "resume_rollback_optimizer_steps"}
    if type(value) is not dict or set(value) not in (base, base | prior):
        raise ValueError("Previous progress keys differ")
    if value["schema"] != SCHEMA or value["phase"] not in PHASES or value["rank"] != 0 or type(value["rank"]) is not int:
        raise ValueError("Previous progress identity differs")
    for name in ("run_id", "prior_run_id"):
        if name in value and (type(value[name]) is not str or re.fullmatch("[0-9a-f]{32}", value[name]) is None):
            raise ValueError("Previous progress run ID differs")
    for name in ("pid", "global_time", "optimizer_steps", "sequence", "resume_global_time", "resume_optimizer_steps", "resume_rollback_global_time", "resume_rollback_optimizer_steps"):
        if name in value:
            _counter(value[name], name)
    if value["pid"] == 0 or value["sequence"] == 0:
        raise ValueError("Previous progress positive identity differs")
    for name in ("run_started_unix_s", "observed_at_unix_s", "elapsed_s", "phase_started_unix_s", "phase_elapsed_s"):
        if type(value[name]) not in (int, float) or not math.isfinite(value[name]) or value[name] < 0:
            raise ValueError("Previous progress time type differs")


class TrainingProgress:
    """One rank-zero writer, one run ID, monotonic completed-work counters.

    A new run starts from the actual checkpoint. If a crash lost unsaved work,
    the receipt explicitly records that rollback instead of claiming it was
    restored. Only changes within the same run must be monotonic.
    """

    def __init__(self, path: Path, *, global_time: int, optimizer_steps: int,
                 interval_seconds: float = 10.0):
        if not math.isfinite(interval_seconds) or interval_seconds <= 0:
            raise ValueError("progress interval must be finite and positive")
        self._output_failed = False
        self.path = Path(path)
        self.interval_seconds = interval_seconds
        self.global_time = _counter(global_time, "global_time")
        self.optimizer_steps = _counter(optimizer_steps, "optimizer_steps")
        self.run_id = uuid.uuid4().hex
        self.started = time.monotonic()
        self.run_started_unix_s = time.time()
        self.phase = "initializing"
        self.phase_started = self.started
        self.phase_started_unix_s = self.run_started_unix_s
        self.last_written = -math.inf
        self.sequence = 0
        self.resume = {"resume_global_time": global_time, "resume_optimizer_steps": optimizer_steps}
        if self.path.exists():
            previous = json.loads(metadata_output.read_text(self.path, kind="progress"))
            _validate_previous(previous)
            if previous.get("schema") != SCHEMA or not isinstance(previous.get("run_id"), str):
                raise ValueError("Existing training progress receipt has an invalid schema/run ID")
            old_global = _counter(previous.get("global_time"), "previous global_time")
            old_updates = _counter(previous.get("optimizer_steps"), "previous optimizer_steps")
            self.resume.update(prior_run_id=previous["run_id"],
                               resume_rollback_global_time=max(0, old_global-global_time),
                               resume_rollback_optimizer_steps=max(0, old_updates-optimizer_steps))
        self.update(force=True)

    def update(self, **kwargs):
        if self._output_failed:
            raise RuntimeError("Metadata output failure is latched; no later progress temporary permitted")
        try:
            return self._update(**kwargs)
        except BaseException:
            self._output_failed = True
            raise

    def _update(self, *, global_time=None, optimizer_steps=None, phase=None, force=False):
        global_time = self.global_time if global_time is None else _counter(global_time, "global_time")
        optimizer_steps = self.optimizer_steps if optimizer_steps is None else _counter(optimizer_steps, "optimizer_steps")
        phase = self.phase if phase is None else phase
        if phase not in PHASES:
            raise ValueError("Unknown training progress phase")
        if global_time < self.global_time or optimizer_steps < self.optimizer_steps:
            raise ValueError("Completed training counters cannot decrease within a run")
        if self.phase in {"finished", "failed"} and phase != self.phase:
            raise ValueError("A terminal training receipt cannot resume within the same run")
        tick = time.monotonic()
        observed = time.time()
        phase_changed = phase != self.phase
        self.global_time, self.optimizer_steps = global_time, optimizer_steps
        if phase_changed:
            self.phase, self.phase_started, self.phase_started_unix_s = phase, tick, observed
        if not force and not phase_changed and tick-self.last_written < self.interval_seconds:
            return False
        receipt = dict(schema=SCHEMA, run_id=self.run_id, pid=os.getpid(), rank=0,
                       run_started_unix_s=self.run_started_unix_s, observed_at_unix_s=observed,
                       elapsed_s=tick-self.started, phase=self.phase,
                       phase_started_unix_s=self.phase_started_unix_s, phase_elapsed_s=tick-self.phase_started,
                       global_time=self.global_time, optimizer_steps=self.optimizer_steps,
                       sequence=self.sequence+1, **self.resume)
        raw = metadata_output.encode_text(self.path, json.dumps(receipt, allow_nan=False) + "\n", kind="progress")
        temporary = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode="wb", dir=self.path.parent,
                                             prefix=".training-progress-", delete=False) as stream:
                temporary = Path(stream.name)
                if stream.write(raw) != len(raw):
                    raise OSError("Short progress metadata write")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            descriptor = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except OSError as exc:
            raise RuntimeError("Cannot publish numeric training progress; refusing to continue without supervision") from exc
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        self.last_written = tick
        self.sequence += 1
        return True


@contextmanager
def progress_phase(reporter, phase, *, global_time, optimizer_steps):
    """Bounded operations keep counters unchanged until work actually finishes."""
    if reporter is None:
        yield
        return
    previous = reporter.phase
    reporter.update(phase=phase, global_time=global_time, optimizer_steps=optimizer_steps, force=True)
    try:
        yield
    except BaseException:
        # Leave this phase in place for the outer failure handler. Reporting a
        # return to training while an operation failed would mislead observers.
        raise
    else:
        reporter.update(phase=previous, force=True)
