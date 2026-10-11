"""CPU-only checks for numeric supervision, durable publication and resume."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from humanoidverse.training.progress import TrainingProgress, progress_phase


class TrainingProgressTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "training_progress.json"

    def read(self):
        return json.loads(self.path.read_text())

    def test_throttle_tracks_completed_counters_without_fabricating_work(self):
        with patch("humanoidverse.training.progress.time.monotonic", return_value=100.) as clock:
            reporter = TrainingProgress(self.path, global_time=800, optimizer_steps=7)
            self.assertEqual(self.read()["global_time"], 800)
            clock.return_value = 104.
            self.assertFalse(reporter.update(global_time=808, optimizer_steps=8))
            self.assertEqual(self.read()["optimizer_steps"], 7)
            self.assertEqual(reporter.optimizer_steps, 8)
            clock.return_value = 110.
            self.assertTrue(reporter.update())
            value = self.read()
            self.assertEqual((value["global_time"], value["optimizer_steps"]), (808, 8))
            self.assertEqual(value["elapsed_s"], 10.)
            clock.return_value = 120.
            reporter.update()
            self.assertEqual((self.read()["global_time"], self.read()["optimizer_steps"]), (808, 8))

    def test_phase_transitions_are_immediate_and_repeated_phase_does_not_reset_age(self):
        with patch("humanoidverse.training.progress.time.monotonic", return_value=10.) as clock:
            reporter = TrainingProgress(self.path, global_time=800, optimizer_steps=7)
            reporter.update(phase="training")
            clock.return_value = 11.
            with progress_phase(reporter, "checkpoint", global_time=800, optimizer_steps=7):
                self.assertEqual(self.read()["phase"], "checkpoint")
                clock.return_value = 20.
                reporter.update(phase="checkpoint", force=True)
                self.assertEqual(self.read()["phase_elapsed_s"], 9.)
                self.assertEqual(self.read()["optimizer_steps"], 7)
            self.assertEqual(self.read()["phase"], "training")

    def test_resume_uses_checkpoint_with_explicit_loss_of_unsaved_work(self):
        previous = TrainingProgress(self.path, global_time=960, optimizer_steps=19)
        restored = TrainingProgress(self.path, global_time=800, optimizer_steps=17)
        value = self.read()
        self.assertNotEqual(previous.run_id, restored.run_id)
        self.assertEqual(value["prior_run_id"], previous.run_id)
        self.assertEqual((value["global_time"], value["optimizer_steps"]), (800, 17))
        self.assertEqual(value["resume_rollback_global_time"], 160)
        self.assertEqual(value["resume_rollback_optimizer_steps"], 2)
        self.assertEqual(value["phase"], "initializing")

    def test_regression_and_invalid_counters_are_rejected_within_one_run(self):
        reporter = TrainingProgress(self.path, global_time=800, optimizer_steps=17)
        for values in ({"global_time": 799}, {"optimizer_steps": 16}, {"global_time": -1},
                       {"optimizer_steps": float("nan")}, {"global_time": True}):
            with self.assertRaises(ValueError):
                reporter.update(**values)
        reporter.update(phase="finished")
        with self.assertRaises(ValueError):
            reporter.update(phase="training")

    def test_atomic_publication_failure_preserves_old_receipt_and_raises(self):
        reporter = TrainingProgress(self.path, global_time=800, optimizer_steps=17)
        before = self.path.read_bytes()
        with patch("humanoidverse.training.progress.os.replace", side_effect=OSError("injected disk failure")):
            with self.assertRaisesRegex(RuntimeError, "refusing to continue"):
                reporter.update(global_time=808, optimizer_steps=18, force=True)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(list(self.path.parent.glob(".training-progress-*")), [])

    def test_failed_operation_does_not_claim_return_to_training(self):
        reporter = TrainingProgress(self.path, global_time=800, optimizer_steps=17)
        reporter.update(phase="training")
        with self.assertRaisesRegex(RuntimeError, "checkpoint failed"):
            with progress_phase(reporter, "checkpoint", global_time=800, optimizer_steps=17):
                raise RuntimeError("checkpoint failed")
        self.assertEqual(self.read()["phase"], "checkpoint")

    def test_workspace_uses_completed_steps_and_resumed_warmup(self):
        from humanoidverse.training.checkpoint_io import ReplayWarmup
        from humanoidverse.training.workspace import Workspace
        reporter = TrainingProgress(self.path, global_time=800_000, optimizer_steps=17)
        workspace = SimpleNamespace(_training_progress=reporter, _optimizer_steps=17,
                                    cfg=SimpleNamespace(num_seed_steps=10240),
                                    _replay_warmup=ReplayWarmup(required_steps=10240))
        Workspace._publish_training_progress(workspace, local_time=100_000, global_time=800_000)
        self.assertEqual(self.read()["phase"], "warmup")
        workspace._replay_warmup.advance(10240)
        workspace._optimizer_steps += 1
        Workspace._publish_training_progress(workspace, local_time=110_240, global_time=881_920)
        self.assertEqual(self.read()["phase"], "training")
        self.assertEqual((self.read()["global_time"], self.read()["optimizer_steps"]), (881_920, 18))
        # Other ranks have no writer and do not need shared-file access at all.
        Workspace._publish_training_progress(SimpleNamespace(_training_progress=None), local_time=0, global_time=0)

    def test_workspace_reports_failed_and_successful_completion(self):
        from humanoidverse.training.workspace import Workspace
        reporter = TrainingProgress(self.path, global_time=800, optimizer_steps=17)
        def fail():
            reporter.update(global_time=808, phase="training")
            raise RuntimeError("optimizer failed")
        workspace = SimpleNamespace(_training_progress=reporter, _optimizer_steps=17, train_online=fail)
        with self.assertRaisesRegex(RuntimeError, "optimizer failed"):
            Workspace.train(workspace)
        self.assertEqual((self.read()["phase"], self.read()["global_time"]), ("failed", 808))
        reporter = TrainingProgress(self.path, global_time=800, optimizer_steps=17)
        def finish():
            reporter.update(global_time=808, optimizer_steps=18, phase="training")
            workspace._optimizer_steps = 18
        workspace._training_progress, workspace.train_online = reporter, finish
        Workspace.train(workspace)
        self.assertEqual((self.read()["phase"], self.read()["global_time"], self.read()["optimizer_steps"]),
                         ("finished", 808, 18))

    def test_workspace_wraps_actual_evaluation_and_checkpoint_operations(self):
        from humanoidverse.training.workspace import Workspace
        reporter = TrainingProgress(self.path, global_time=800, optimizer_steps=17)
        reporter.update(phase="training")
        workspace = SimpleNamespace(_training_progress=reporter, _optimizer_steps=17)
        def evaluate(*args):
            self.assertEqual(self.read()["phase"], "evaluation")
            return {"measured": True}
        def save(*args, **kwargs):
            self.assertEqual(self.read()["phase"], "checkpoint")
            self.assertEqual((self.read()["global_time"], self.read()["optimizer_steps"]), (800, 17))
        with patch.object(Workspace, "_run_evaluation", side_effect=evaluate):
            self.assertEqual(Workspace.eval(workspace, 800, {}), {"measured": True})
        self.assertEqual(self.read()["phase"], "training")
        with patch.object(Workspace, "_save_checkpoint", side_effect=save):
            Workspace.save(workspace, local_time=100, global_time=800, optimizer_steps=17, replay_buffer={})
        self.assertEqual(self.read()["phase"], "training")


if __name__ == "__main__":
    unittest.main()
