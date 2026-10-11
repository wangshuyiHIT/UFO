"""Opt-in immutable checkpoints, with bounded distributed error coordination.

Only the published ``checkpoint`` symlink is a resume candidate. A complete but
unpublished directory is never promoted automatically after an interrupted save.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import shutil
import uuid
import warnings
from collections.abc import Mapping
from datetime import timedelta
from dataclasses import dataclass
from pathlib import Path

GIB = 1024 ** 3
GENERATIONS = "checkpoint_generations"
REQUIRED = ("config.json", "init_kwargs.json", "optimizers.pth", "model/config.json",
            "model/model.safetensors", "train_status.json")


@dataclass
class ReplayWarmup:
    """Count new local environment steps without changing saved training clocks."""
    required_steps: int = 0
    collected_steps: int = 0

    @classmethod
    def for_resume(cls, *, transactional: bool, resumed: bool, replay_loaded: bool, seed_steps: int):
        return cls(max(0, int(seed_steps)) if transactional and resumed and not replay_loaded else 0)

    def advance(self, local_env_steps: int) -> None:
        if local_env_steps < 0:
            raise ValueError("Collected environment steps must be nonnegative")
        self.collected_steps += local_env_steps

    @property
    def ready(self) -> bool:
        return self.collected_steps >= self.required_steps


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_generation(path: Path, *, full_hashes: bool = False) -> dict:
    manifest = json.loads((path / "COMMITTED.json").read_text())
    if manifest.get("version") != 1 or not isinstance(manifest.get("files"), dict):
        raise RuntimeError(f"Invalid checkpoint manifest: {path}")
    for relative in REQUIRED:
        if relative not in manifest["files"]:
            raise RuntimeError(f"Incomplete checkpoint: missing {relative}")
    for relative, record in manifest["files"].items():
        item = path / relative
        if Path(relative).is_absolute() or ".." in Path(relative).parts or item.is_symlink():
            raise RuntimeError("Unsafe checkpoint manifest path")
        if not item.is_file() or item.stat().st_size != record["bytes"]:
            raise RuntimeError(f"Checkpoint file missing or truncated: {item}")
        if full_hashes or record["bytes"] < 1024 * 1024:
            if file_digest(item) != record["sha256"]:
                raise RuntimeError(f"Checkpoint digest mismatch: {item}")
    return manifest


def resolve_checkpoint(work_dir: Path, *, transactional: bool) -> Path:
    """Keep legacy reads; reject mixing in-place and transactional writers."""
    checkpoint = work_dir / "checkpoint"
    if checkpoint.is_symlink():
        if not transactional:
            raise RuntimeError("This run uses transactional checkpoints; enable --transactional-checkpoints to resume")
        target = checkpoint.resolve(strict=True)
        if target.parent != (work_dir / GENERATIONS).resolve() or not target.name.startswith("generation-"):
            raise RuntimeError("Checkpoint symlink is outside the run's generation directory")
        validate_generation(target)
        return target
    if checkpoint.exists() and transactional:
        raise RuntimeError("Refusing to overwrite a legacy checkpoint directory; use a new work_dir for transactional checkpoints")
    # Older Renaissance generations used checkpoint/CURRENT. Do not migrate them
    # implicitly; ordinary legacy runs still read their existing checkpoint path.
    pointer = checkpoint / "CURRENT"
    if pointer.is_file():
        generation = pointer.read_text().strip()
        if not generation or Path(generation).name != generation:
            raise RuntimeError("Invalid legacy CURRENT checkpoint pointer")
        target = checkpoint / "generations" / generation
        if not (target / "COMMITTED.json").is_file():
            raise RuntimeError("Incomplete legacy CURRENT checkpoint generation")
        return target
    return checkpoint


class CheckpointStore:
    def __init__(self, work_dir: Path, *, reserve_gib: float = 4.0):
        if not math.isfinite(reserve_gib) or reserve_gib < 0:
            raise ValueError("checkpoint reserve must be finite and nonnegative")
        self.work_dir = Path(work_dir)
        self.root = self.work_dir / GENERATIONS
        self.reserve_bytes = math.ceil(reserve_gib * GIB)
        self.stage = None
        self.lock = None
        self.previous = None

    def begin(self, estimated_bytes: int) -> Path:
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.lock = (self.work_dir / ".checkpoint.lock").open("a")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        current = resolve_checkpoint(self.work_dir, transactional=True)
        if current.is_dir():
            self.previous = current.name
            estimated_bytes = max(estimated_bytes, sum(p.stat().st_size for p in current.rglob("*") if p.is_file()))
        required = math.ceil(max(estimated_bytes, 0) * 1.2) + self.reserve_bytes
        available = shutil.disk_usage(self.work_dir).free
        if available < required:
            raise OSError(f"Checkpoint needs {required} free bytes (one generation plus reserve), has {available}; previous generation retained")
        self.root.mkdir(exist_ok=True)
        self.stage = self.root / ("generation-" + uuid.uuid4().hex)
        self.stage.mkdir()
        return self.stage

    def commit(self, *, expected_buffers: tuple[str, ...] = (), fault_hook=None) -> Path:
        stage = self.stage
        if stage is None:
            raise RuntimeError("Checkpoint transaction has not started")
        for relative in (*REQUIRED, *expected_buffers):
            if not (stage / relative).is_file():
                raise RuntimeError(f"Missing staged checkpoint artifact: {relative}")
        files = {}
        for item in sorted(stage.rglob("*")):
            if item.is_symlink():
                raise RuntimeError("Symlinks are not permitted inside a checkpoint generation")
            if item.is_file():
                files[str(item.relative_to(stage))] = {"bytes": item.stat().st_size, "sha256": file_digest(item)}
        manifest = {"version": 1, "previous": self.previous, "files": files}
        (stage / "COMMITTED.json").write_text(json.dumps(manifest, indent=2) + "\n")
        for item in stage.rglob("*"):
            if item.is_file():
                with item.open("rb") as stream:
                    os.fsync(stream.fileno())
        for directory in sorted([stage, *(p for p in stage.rglob("*") if p.is_dir())], key=lambda p: len(p.parts), reverse=True):
            _fsync_dir(directory)
        _fsync_dir(self.root)
        pointer = self.work_dir / (".checkpoint-pointer-" + uuid.uuid4().hex)
        try:
            pointer.symlink_to(Path(GENERATIONS) / stage.name)
            if fault_hook:
                fault_hook("before_publish")
            os.replace(pointer, self.work_dir / "checkpoint")
        finally:
            pointer.unlink(missing_ok=True)
        # Publication is the commit point. Cleanup/directory-sync failures after
        # it must not report a failed save or remove the now-current generation.
        self.stage = None
        try:
            _fsync_dir(self.work_dir)
            for candidate in self.root.iterdir():
                if (candidate.is_dir() and not candidate.is_symlink()
                        and candidate.name.startswith("generation-")
                        and candidate.name not in {stage.name, self.previous}):
                    shutil.rmtree(candidate)
            _fsync_dir(self.root)
        except OSError as exc:
            warnings.warn(f"Checkpoint committed; durability/retention cleanup needs attention: {exc}")
        return stage

    def close(self) -> None:
        if self.stage is not None:
            shutil.rmtree(self.stage, ignore_errors=True)
            self.stage = None
        if self.lock is not None:
            self.lock.close()
            self.lock = None


class CheckpointCoordinator:
    """A separate CPU group bounds checkpoint waits independently of NCCL."""
    def __init__(self, *, world_size: int, timeout_seconds: float = 300):
        import torch.distributed as dist
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("checkpoint timeout must be finite and positive")
        self.group = None
        self.world_size = world_size
        self.rank = 0
        if world_size > 1:
            if not dist.is_initialized() or dist.get_world_size() != world_size:
                raise RuntimeError("Transactional distributed saves require an initialized matching process group")
            self.rank = dist.get_rank()
            self.group = dist.new_group(backend="gloo", timeout=timedelta(seconds=timeout_seconds))

    def gather(self, value):
        if self.group is None:
            return [value]
        import torch.distributed as dist
        gathered = [None] * self.world_size
        dist.all_gather_object(gathered, value, group=self.group)
        return gathered

    def stage(self, name, callback):
        value = error = None
        try:
            value = callback()
        except Exception as exc:
            error = f"rank {self.rank}: {type(exc).__name__}: {exc}"
        results = self.gather({"error": error, "value": value})
        errors = [item["error"] for item in results if item["error"]]
        if errors:
            raise RuntimeError(f"Checkpoint {name} failed; previous generation retained: " + "; ".join(errors))
        return [item["value"] for item in results]


def tensor_bytes(value) -> int:
    import torch
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, Mapping):
        return sum(tensor_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(tensor_bytes(item) for item in value)
    return 0


def model_digest(model) -> str:
    import torch
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(f"{name}:{tensor.dtype}:{tuple(tensor.shape)}".encode())
        # One tensor at a time, avoiding another full model allocation on GPU.
        digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def transactional_save(workspace, *, train_status: dict, replay_buffer: dict, fault_hook=None) -> Path:
    coordinator = workspace._checkpoint_coordinator
    rank = coordinator.rank
    cfg = workspace.cfg
    store = CheckpointStore(workspace.work_dir, reserve_gib=cfg.checkpoint_reserve_gib) if rank == 0 else None
    try:
        def describe():
            size = (tensor_bytes(workspace.agent._model.state_dict()) + tensor_bytes(workspace.agent.optimizer_dict)) if rank == 0 else 0
            if cfg.checkpoint_buffer:
                size += tensor_bytes(replay_buffer["train"].storage)
            return {"bytes": size + (16 * 1024 * 1024 if rank == 0 else 0), "model_sha256": model_digest(workspace.agent._model)}
        descriptions = coordinator.stage("state check", describe)
        if len({item["model_sha256"] for item in descriptions}) != 1:
            raise RuntimeError("Checkpoint model state differs between ranks; previous generation retained")
        roots = coordinator.stage("disk preflight", lambda: str(store.begin(sum(item["bytes"] for item in descriptions))) if rank == 0 else None)
        stage = Path(roots[0])

        def write():
            if rank == 0:
                workspace.agent.save(str(stage))
            if cfg.checkpoint_buffer:
                replay_buffer["train"].save(workspace._checkpoint_buffer_path(stage))
            if rank == 0:
                (stage / "train_status.json").write_text(json.dumps(train_status, indent=2))
                (stage / "distributed_sync.json").write_text(json.dumps({"world_size": coordinator.world_size, "model_sha256": descriptions[0]["model_sha256"], "exact_model_match": True}, indent=2))
                (stage / "effective_train_config.json").write_text(cfg.model_dump_json(indent=2))
            if fault_hook:
                fault_hook("after_write")
        coordinator.stage("write", write)
        expected = ()
        if cfg.checkpoint_buffer:
            expected = tuple(f"buffers/train_rank_{worker}/buffer.hdf5" for worker in range(coordinator.world_size)) if coordinator.world_size > 1 else ("buffers/train/buffer.hdf5",)
        coordinator.stage("publication", lambda: str(store.commit(expected_buffers=expected, fault_hook=fault_hook)) if rank == 0 else None)
        return stage
    finally:
        if store is not None:
            store.close()
