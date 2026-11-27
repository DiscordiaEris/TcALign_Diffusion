import unittest
from types import SimpleNamespace
from pathlib import Path
import tempfile
from unittest.mock import Mock
import torch
import math

from trainers.base_trainer import BaseTrainer
from trainers.stage2_trainer import Stage2Trainer


class IdentitySelectionStateTest(unittest.TestCase):
    def test_invalid_saved_best_scores_preserve_an_existing_finite_threshold(self):
        for score in (None, math.nan, math.inf, -math.inf):
            with self.subTest(score=score):
                trainer = BaseTrainer.__new__(BaseTrainer)
                trainer._best_sampled_identity_score = 0.77
                trainer.restore_identity_validation({"identity_validation": {"best_score": score}})
                self.assertEqual(trainer._best_sampled_identity_score, 0.77)

    def test_unmeasured_checkpoint_does_not_create_an_invalid_best_threshold(self):
        trainer = BaseTrainer.__new__(BaseTrainer)
        trainer.restore_identity_validation({"identity_validation": {"best_score": None}})
        self.assertFalse(hasattr(trainer, "_best_sampled_identity_score"))

    def test_existing_invalid_reference_does_not_break_loading_a_finite_disk_best(self):
        trainer = Stage2Trainer.__new__(Stage2Trainer)
        trainer._best_sampled_identity_score = None
        with tempfile.TemporaryDirectory() as directory:
            trainer.config = SimpleNamespace(identity_validation_subjects=6, checkpoint_dir=directory)
            torch.save({"identity_validation": {"best_score": 0.85}},
                       Path(directory) / "best_identity_checkpoint.pth")
            trainer.initialize_identity_validation()
        self.assertEqual(trainer._best_sampled_identity_score, 0.85)

    def test_best_identity_reference_survives_resume_and_weights_only_initialization(self):
        trainer = BaseTrainer.__new__(BaseTrainer)
        checkpoint = {"identity_validation": {"best_score": .78,
                                               "latest": {"identity_score": .76}}}
        trainer.restore_identity_validation(checkpoint)
        self.assertEqual(trainer._best_sampled_identity_score, .78)
        self.assertEqual(trainer._last_sampled_identity_metrics["identity_score"], .76)

    def test_legacy_checkpoint_does_not_erase_existing_selection_state(self):
        trainer = BaseTrainer.__new__(BaseTrainer)
        trainer._best_sampled_identity_score = .77
        trainer.restore_identity_validation({"global_step": 2})
        self.assertEqual(trainer._best_sampled_identity_score, .77)

    def test_missing_baseline_is_measured_once_without_advancing_training_rng(self):
        trainer = Stage2Trainer.__new__(Stage2Trainer)
        trainer.config = SimpleNamespace(identity_validation_subjects=6)
        trainer.device = "cpu"
        trainer.model = torch.nn.Linear(1, 1)
        calls = []
        def measure():
            calls.append(trainer.model.training)
            torch.randn(7)
            trainer._best_sampled_identity_score = .8
        trainer.validate_sampled_identity = measure
        state = torch.random.get_rng_state().clone()
        trainer.initialize_identity_validation()
        trainer.initialize_identity_validation()
        self.assertEqual(calls, [False])
        self.assertTrue(trainer.model.training)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))

    def test_resume_latest_cannot_lower_the_existing_best_checkpoint_threshold(self):
        trainer = Stage2Trainer.__new__(Stage2Trainer)
        trainer._best_sampled_identity_score = .7
        with tempfile.TemporaryDirectory() as directory:
            trainer.config = SimpleNamespace(identity_validation_subjects=6, checkpoint_dir=directory)
            torch.save({"identity_validation": {"best_score": .85}},
                       Path(directory) / "best_identity_checkpoint.pth")
            trainer.initialize_identity_validation()
        self.assertEqual(trainer._best_sampled_identity_score, .85)

    def test_yaml_resume_precedes_identity_selection_initialization(self):
        trainer = BaseTrainer.__new__(BaseTrainer)
        trainer.config = SimpleNamespace(resume_from_checkpoint="legacy.pth", num_epochs=0)
        trainer.epoch = 0
        trainer.logger = Mock()
        calls = []
        trainer.load_checkpoint = lambda path: calls.append("resume")
        trainer.on_training_start = lambda: calls.append("identity_baseline")
        trainer.train()
        self.assertEqual(calls, ["resume", "identity_baseline"])

    def test_resume_of_stopped_calibration_does_not_take_another_epoch(self):
        trainer = BaseTrainer.__new__(BaseTrainer)
        trainer.config = SimpleNamespace(resume_from_checkpoint=None, num_epochs=100, checkpoint_dir="unused")
        trainer.epoch = 5
        trainer.stop_training = True
        trainer.logger = Mock()
        trainer.on_training_start = Mock()
        trainer.train_one_epoch = Mock(return_value=0.)
        trainer.save_checkpoint = Mock()
        trainer.train()
        trainer.train_one_epoch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
