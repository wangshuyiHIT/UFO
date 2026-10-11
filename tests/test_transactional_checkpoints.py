from __future__ import annotations

import json
import multiprocessing
import sys
import tempfile
import time
import unittest
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist

from humanoidverse.training.checkpoint_io import (
    CheckpointCoordinator, CheckpointStore, GENERATIONS, ReplayWarmup,
    resolve_checkpoint, transactional_save, validate_generation,
)


class FakeAgent:
    def __init__(self):
        self._model = torch.nn.Linear(2, 1)
        with torch.no_grad():
            self._model.weight.fill_(2)
            self._model.bias.fill_(3)
        self.optimizer = torch.optim.Adam(self._model.parameters())
        self._model(torch.ones(1, 2)).sum().backward()
        self.optimizer.step()
        self.fail = False

    @property
    def optimizer_dict(self):
        return {"optimizer": self.optimizer.state_dict()}

    def save(self, destination):
        destination = Path(destination)
        (destination / "model").mkdir(exist_ok=True)
        torch.save(self.optimizer_dict, destination / "optimizers.pth")
        if self.fail:
            raise OSError("injected full disk after optimizer write")
        torch.save(self._model.state_dict(), destination / "model/model.safetensors")
        for relative in ("config.json", "model/config.json", "init_kwargs.json"):
            (destination / relative).write_text("{}")


class FakeBuffer:
    def __init__(self):
        self.storage = {"observations": torch.ones(2, 3)}

    def save(self, destination):
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "buffer.hdf5").write_bytes(b"fake replay bytes")


class FakeConfig(SimpleNamespace):
    def model_dump_json(self, **kwargs):
        return json.dumps(vars(self), **kwargs)


def workspace(root, *, world_size=1, rank=0, replay=False):
    cfg = FakeConfig(checkpoint_reserve_gib=0, checkpoint_buffer=replay)
    return SimpleNamespace(
        cfg=cfg, work_dir=Path(root), agent=FakeAgent(),
        _checkpoint_coordinator=CheckpointCoordinator(world_size=world_size, timeout_seconds=5),
        _checkpoint_buffer_path=lambda stage: stage / "buffers" / (f"train_rank_{rank}" if world_size > 1 else "train"),
    )


def save(ws, step, hook=None):
    return transactional_save(ws, train_status={"global_time": step, "optimizer_steps": step},
                              replay_buffer={"train": FakeBuffer()}, fault_hook=hook)


def distributed_worker(rank, rendezvous, root):
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank,
                            world_size=2, timeout=timedelta(seconds=15))
    try:
        ws = workspace(root, world_size=2, rank=rank, replay=True)
        save(ws, 1)
        original = (Path(root) / "checkpoint").resolve()
        errors = []
        for mode in ("rank1_replay", "rank0_model", "rank0_publish", "diverged"):
            def hook(stage):
                if mode == "rank1_replay" and rank == 1 and stage == "after_write":
                    raise OSError("injected rank 1 replay failure")
                if mode == "rank0_publish" and rank == 0 and stage == "before_publish":
                    raise OSError("injected publication failure")
            ws.agent.fail = mode == "rank0_model" and rank == 0
            if mode == "diverged" and rank == 1:
                with torch.no_grad():
                    ws.agent._model.bias.add_(1)
            started = time.monotonic()
            try:
                save(ws, 2, hook)
                raise AssertionError("Injected failure unexpectedly published")
            except RuntimeError as exc:
                errors.append({"mode": mode, "error": str(exc), "seconds": time.monotonic() - started})
            assert (Path(root) / "checkpoint").resolve() == original
        (Path(root) / f"rank_{rank}_result.json").write_text(json.dumps(errors))
    finally:
        dist.destroy_process_group()


def stalled_worker(rank, rendezvous, root):
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank,
                            world_size=2, timeout=timedelta(seconds=15))
    try:
        coordinator = CheckpointCoordinator(world_size=2, timeout_seconds=0.4)
        started = time.monotonic()
        def callback():
            if rank == 1:
                time.sleep(1.2)
        try:
            coordinator.stage("stalled writer", callback)
            raise AssertionError("A stalled rank exceeded the checkpoint wait without an error")
        except RuntimeError:
            (Path(root) / f"rank_{rank}_timeout.json").write_text(json.dumps({"seconds": time.monotonic() - started}))
    finally:
        dist.destroy_process_group()


class TransactionalCheckpointTest(unittest.TestCase):
    def test_replayless_resume_collects_new_steps_without_resetting_clocks(self):
        warmup = ReplayWarmup.for_resume(transactional=True, resumed=True, replay_loaded=False, seed_steps=50_000)
        self.assertFalse(warmup.ready)
        warmup.advance(49_999)
        self.assertFalse(warmup.ready)
        warmup.advance(1)
        self.assertTrue(warmup.ready)

    def test_workspace_config_cli_publication_and_resume_counters(self):
        from humanoidverse.train import build_ufo_mjlab_config, parse_args
        from humanoidverse.training.workspace import Workspace, create_agent_or_load_checkpoint
        with patch.object(sys, "argv", ["train.py", "--transactional-checkpoints", "--no-checkpoint-buffer", "--checkpoint-reserve-gib", "0", "--checkpoint-timeout-seconds", "17"]):
            args = parse_args()
        with tempfile.TemporaryDirectory() as directory:
            ws = workspace(directory)
            ws.cfg = build_ufo_mjlab_config(device="cpu", work_dir=directory, num_envs=1, num_env_steps=100,
                                            seed=0, use_wandb=False, wandb_run_name=None, smoke=True,
                                            transactional_checkpoints=args.transactional_checkpoints,
                                            checkpoint_buffer=args.checkpoint_buffer,
                                            checkpoint_reserve_gib=args.checkpoint_reserve_gib,
                                            checkpoint_timeout_seconds=args.checkpoint_timeout_seconds)
            ws._replay_warmup = ReplayWarmup(required_steps=50_000, collected_steps=123)
            Workspace.save(ws, local_time=1_000_000, global_time=1_000_000, optimizer_steps=4312, replay_buffer={})
            published = resolve_checkpoint(Path(directory), transactional=True)
            with patch.object(ws.cfg.agent.object_class, "load", return_value=ws.agent) as restore:
                restored, config, status = create_agent_or_load_checkpoint(Path(directory), ws.cfg, {})
            restore.assert_called_once_with(published, device="cpu")
            self.assertIs(restored, ws.agent)
            self.assertEqual(status["local_time"], 1_000_000)
            self.assertEqual(status["global_time"], 1_000_000)
            self.assertEqual(status["optimizer_steps"], 4312)
            self.assertEqual(config.checkpoint_timeout_seconds, 17)
            saved_status = json.loads((published / "train_status.json").read_text())
            self.assertFalse(saved_status["replay_buffer_saved"])
            self.assertEqual(saved_status["resume_replay_warmup"]["collected_local_steps"], 123)

    def test_loaded_replay_fresh_run_and_default_mode_do_not_add_warmup(self):
        for transactional, resumed, replay_loaded in ((True, True, True), (True, False, False), (False, True, False)):
            warmup = ReplayWarmup.for_resume(transactional=transactional, resumed=resumed, replay_loaded=replay_loaded, seed_steps=50_000)
            self.assertTrue(warmup.ready)
            self.assertEqual(warmup.required_steps, 0)

    def test_final_save_preserves_completed_steps_without_duplicate_generation(self):
        from humanoidverse.training.workspace import Workspace
        from unittest.mock import Mock
        ws = SimpleNamespace(cfg=SimpleNamespace(transactional_checkpoints=True),
                             _optimizer_steps=12, _last_checkpoint_position=(64, 512, 10), save=Mock())
        Workspace._save_final_checkpoint(ws, local_time=80, global_time=640, replay_buffer={})
        ws.save.assert_called_once_with(local_time=80, global_time=640, optimizer_steps=12, replay_buffer={})
        ws.save.reset_mock()
        ws._last_checkpoint_position = (80, 640, 12)
        Workspace._save_final_checkpoint(ws, local_time=80, global_time=640, replay_buffer={})
        ws.save.assert_not_called()
        ws.cfg.transactional_checkpoints = False
        Workspace._save_final_checkpoint(ws, local_time=96, global_time=768, replay_buffer={})
        ws.save.assert_not_called()

    def test_complete_generation_retention_and_restore_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            ws = workspace(directory)
            first = save(ws, 10)
            second = save(ws, 20)
            # A complete-looking orphan must not displace the actual predecessor.
            orphan = Path(directory) / GENERATIONS / "generation-unpublished"
            orphan.mkdir()
            (orphan / "train_status.json").write_text('{"global_time":999}')
            self.assertEqual(resolve_checkpoint(Path(directory), transactional=True), second)
            third = save(ws, 30)
            self.assertEqual(set((Path(directory) / GENERATIONS).iterdir()), {second, third})
            self.assertFalse(first.exists())
            self.assertEqual(json.loads((third / "train_status.json").read_text())["global_time"], 30)
            restored = torch.load(third / "model/model.safetensors", weights_only=True)
            self.assertTrue(torch.equal(restored["weight"], ws.agent._model.weight))
            restored_optimizer = torch.load(third / "optimizers.pth", weights_only=True)
            self.assertEqual(restored_optimizer["optimizer"]["state"][0]["step"], 1)
            validate_generation(third, full_hashes=True)

    def test_mid_save_and_publication_failures_preserve_previous(self):
        with tempfile.TemporaryDirectory() as directory:
            ws = workspace(directory)
            original = save(ws, 10)
            ws.agent.fail = True
            with self.assertRaisesRegex(RuntimeError, "injected full disk"):
                save(ws, 20)
            ws.agent.fail = False
            def fail(stage):
                if stage == "before_publish":
                    raise OSError("injected publication failure")
            with self.assertRaisesRegex(RuntimeError, "publication failure"):
                save(ws, 30, fail)
            self.assertEqual(resolve_checkpoint(Path(directory), transactional=True), original)
            self.assertEqual(list((Path(directory) / GENERATIONS).iterdir()), [original])

    def test_failed_train_status_write_preserves_previous(self):
        with tempfile.TemporaryDirectory() as directory:
            ws = workspace(directory)
            original = save(ws, 1)
            real_write = Path.write_text
            def fail(path, *args, **kwargs):
                if path.name == "train_status.json":
                    raise OSError("injected status write failure")
                return real_write(path, *args, **kwargs)
            with patch.object(Path, "write_text", fail), self.assertRaisesRegex(RuntimeError, "status write failure"):
                save(ws, 2)
            self.assertEqual(resolve_checkpoint(Path(directory), transactional=True), original)

    def test_low_disk_and_actual_generation_size(self):
        with tempfile.TemporaryDirectory() as directory:
            ws = workspace(directory)
            original = save(ws, 1)
            old_bytes = sum(p.stat().st_size for p in original.rglob("*") if p.is_file())
            store = CheckpointStore(Path(directory), reserve_gib=0)
            with patch("humanoidverse.training.checkpoint_io.shutil.disk_usage", return_value=SimpleNamespace(free=old_bytes)):
                try:
                    with self.assertRaisesRegex(OSError, "one generation plus reserve"):
                        store.begin(1)
                finally:
                    store.close()
            self.assertEqual(resolve_checkpoint(Path(directory), transactional=True), original)

    def test_legacy_read_and_write_modes_are_not_mixed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "checkpoint").mkdir()
            marker = root / "checkpoint" / "old_weights"
            marker.write_bytes(b"untouched")
            self.assertEqual(resolve_checkpoint(root, transactional=False), root / "checkpoint")
            with self.assertRaisesRegex(RuntimeError, "legacy checkpoint"):
                resolve_checkpoint(root, transactional=True)
            self.assertEqual(marker.read_bytes(), b"untouched")
        with tempfile.TemporaryDirectory() as directory:
            save(workspace(directory), 1)
            with self.assertRaisesRegex(RuntimeError, "enable --transactional"):
                resolve_checkpoint(Path(directory), transactional=False)

    def test_corrupt_or_dangling_pointer_never_falls_back_to_newest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = save(workspace(directory), 1)
            (target / "optimizers.pth").write_bytes(b"truncated")
            with self.assertRaisesRegex(RuntimeError, "truncated"):
                resolve_checkpoint(root, transactional=True)
            (root / "checkpoint").unlink()
            (root / "checkpoint").symlink_to("checkpoint_generations/generation-missing")
            with self.assertRaises(FileNotFoundError):
                resolve_checkpoint(root, transactional=True)

    def test_exclusive_writer_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            first = CheckpointStore(Path(directory), reserve_gib=0)
            other = CheckpointStore(Path(directory), reserve_gib=0)
            try:
                first.begin(1)
                with self.assertRaises(BlockingIOError):
                    other.begin(1)
            finally:
                other.close()
                first.close()

    def test_two_ranks_fail_together_without_barrier_hang(self):
        with tempfile.TemporaryDirectory() as directory:
            context = multiprocessing.get_context("spawn")
            children = [context.Process(target=distributed_worker, args=(rank, str(Path(directory) / "rendezvous"), directory)) for rank in range(2)]
            for child in children:
                child.start()
            try:
                deadline = time.monotonic() + 40
                for child in children:
                    child.join(max(0, deadline - time.monotonic()))
                self.assertEqual([child.exitcode for child in children], [0, 0])
                results = [json.loads((Path(directory) / f"rank_{rank}_result.json").read_text()) for rank in range(2)]
                for rank_results in results:
                    self.assertEqual(len(rank_results), 4)
                    self.assertTrue(all(record["seconds"] < 5 for record in rank_results))
                self.assertEqual([r["error"] for r in results[0]], [r["error"] for r in results[1]])
            finally:
                for child in children:
                    if child.is_alive():
                        child.terminate()
                    child.join()

    def test_checkpoint_group_timeout_bounds_a_stalled_rank(self):
        with tempfile.TemporaryDirectory() as directory:
            context = multiprocessing.get_context("spawn")
            children = [context.Process(target=stalled_worker, args=(rank, str(Path(directory) / "rendezvous"), directory)) for rank in range(2)]
            for child in children:
                child.start()
            try:
                deadline = time.monotonic() + 20
                for child in children:
                    child.join(max(0, deadline - time.monotonic()))
                self.assertEqual([child.exitcode for child in children], [0, 0])
                rank0 = json.loads((Path(directory) / "rank_0_timeout.json").read_text())
                self.assertLess(rank0["seconds"], 2)
            finally:
                for child in children:
                    if child.is_alive():
                        child.terminate()
                    child.join()


if __name__ == "__main__":
    unittest.main()
