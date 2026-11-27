import logging
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from trainers.stage2_trainer import Stage2Trainer


def metrics(score=0.8, expression=0.4, progress=0.8, direction_hits=6):
    return {"aggregate": {
        "subjects_expected": 6, "subjects_selected": 6,
        "identity_expected": 18, "identity_valid": 18, "identity_score": score,
        "pose_expected": 12, "pose_valid": 12, "pose_direction_hits": 12,
        "pose_yaw_abs_error_deg": 5.0,
        "exposure_expected": 12, "exposure_valid": 12, "exposure_error": 0.01,
        "expression_expected": 6, "expression_valid": 6,
        "expression_error_ratio": expression,
        "expression_progress": progress, "expression_progress_valid": 6,
        "expression_direction_hits": direction_hits,
    }, "rows": []}


class SampleOnlyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gain = nn.Parameter(torch.zeros(()))

    def forward(self, *args, **kwargs):
        raise AssertionError("Calibration validation must skip single-step denoising")

    def checkpoint_state(self):
        return self.state_dict()

    def load_checkpoint_state(self, state):
        self.load_state_dict(state)


class IdentityCalibrationTrainerTest(unittest.TestCase):
    def test_partial_latest_policy_recovers_original_arcface_floor_from_best(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = self.trainer(directory, identity_loss_backend="arcface")
            baseline = metrics()
            baseline["aggregate"].update(arcface_raw_valid=18, arcface_raw_cosine=.75)
            with patch("utils.identity_validation.evaluate_sampled_identity", return_value=baseline):
                trainer.initialize_identity_validation()
            latest = torch.load(Path(directory) / "latest_checkpoint.pth", weights_only=True)
            del latest["identity_validation"]["calibration"]["min_arcface_score"]
            restored = self.trainer(directory, identity_loss_backend="arcface")
            restored.restore_identity_validation(latest)
            with patch("utils.identity_validation.evaluate_sampled_identity", side_effect=AssertionError("remeasured")):
                restored.initialize_identity_validation()
            self.assertEqual(restored._identity_calibration_min_arcface_score, .75)

    def test_arcface_baseline_survives_resume_and_rejects_disagreement(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = self.trainer(directory, identity_loss_backend="arcface")
            baseline = metrics()
            baseline["aggregate"].update(arcface_raw_valid=18, arcface_raw_cosine=.75)
            with patch("utils.identity_validation.evaluate_sampled_identity", return_value=baseline):
                trainer.initialize_identity_validation()
            disagreement = metrics(score=.81)
            disagreement["aggregate"].update(arcface_raw_valid=18, arcface_raw_cosine=.74)
            with patch("utils.identity_validation.evaluate_sampled_identity", return_value=disagreement):
                trainer.evaluate()
            self.assertEqual(trainer._best_sampled_identity_score, .8)
            path = Path(directory) / "latest_checkpoint.pth"
            saved = torch.load(path, weights_only=True)
            self.assertEqual(saved["identity_validation"]["calibration"]["min_arcface_score"], .75)
            # This fixture uses SimpleNamespace; real dataclass configs are saved automatically.
            saved["training_config"] = {"identity_loss_backend": "arcface"}
            torch.save(saved, path)
            restored = self.trainer(directory, identity_loss_backend="arcface")
            restored.load_checkpoint(path)
            with patch("utils.identity_validation.evaluate_sampled_identity", side_effect=AssertionError("remeasured")):
                restored.initialize_identity_validation()
            self.assertEqual(restored._identity_calibration_min_arcface_score, .75)

    def trainer(self, directory, **overrides):
        trainer = Stage2Trainer.__new__(Stage2Trainer)
        config = {
            "identity_gain_only": True, "identity_validation_subjects": 6,
            "learning_rate": 0.01, "unet_learning_rate": 1e-6,
            "weight_decay": 0.0001, "beta1": 0.9, "beta2": 0.999,
            "seed": 42, "output_dir": str(directory), "checkpoint_dir": str(directory),
        }
        trainer.config = SimpleNamespace(**{**config, **overrides})
        trainer.device = "cpu"
        trainer.model = SampleOnlyModel()
        trainer.val_dataloader = SimpleNamespace(dataset=object())
        trainer.optimizer = torch.optim.AdamW(trainer.model.parameters())
        trainer.scaler = torch.amp.GradScaler("cpu", enabled=False)
        trainer.logger = logging.getLogger("identity_calibration_trainer_test")
        trainer.global_step = 0
        trainer.epoch = 0
        trainer.generate_examples = self.forbid_examples
        return trainer

    @staticmethod
    def forbid_examples():
        raise AssertionError("Calibration validation must skip the old example path")

    def evaluate(self, trainer, score=0.8, expression=0.4, progress=0.8, direction_hits=6):
        with patch("utils.identity_validation.evaluate_sampled_identity",
                   return_value=metrics(score, expression, progress, direction_hits)):
            trainer.evaluate()

    def initialize(self, trainer, score=0.8, expression=0.4, progress=0.8, direction_hits=6):
        with patch("utils.identity_validation.evaluate_sampled_identity",
                   return_value=metrics(score, expression, progress, direction_hits)):
            trainer.initialize_identity_validation()

    def test_identity_gain_uses_full_learning_rate_inside_native_attention(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = self.trainer(directory)
            model = nn.Module()
            model.diffusion_unet = nn.Module()
            model.diffusion_unet.unet = nn.Module()
            model.diffusion_unet.unet.attn2 = nn.Module()
            attention = model.diffusion_unet.unet.attn2
            attention.weight = nn.Parameter(torch.ones(2, 2))
            attention.processor = nn.Module()
            attention.processor.identity_gain_logit = nn.Parameter(torch.zeros(()))
            attention.processor.identity_refinement = nn.Linear(2, 2, bias=False)
            trainer.model = model
            optimizer = trainer.configure_optimizers()
            groups = {id(p): group for group in optimizer.param_groups for p in group["params"]}
            self.assertEqual(groups[id(attention.processor.identity_gain_logit)]["lr"], 0.01)
            self.assertEqual(groups[id(attention.processor.identity_gain_logit)]["weight_decay"], 0)
            self.assertEqual(groups[id(attention.processor.identity_refinement.weight)]["lr"], 0.01)
            self.assertEqual(groups[id(attention.weight)]["lr"], 1e-6)

    def test_gain_evaluation_only_samples_and_preserves_rng_and_model_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = self.trainer(directory)
            seen_modes = []

            def sample(*args, **kwargs):
                seen_modes.append(trainer.model.training)
                torch.randn(7)
                return metrics()

            for training in (True, False):
                trainer.model.train(training)
                before = torch.random.get_rng_state().clone()
                with patch("utils.identity_validation.evaluate_sampled_identity", side_effect=sample):
                    trainer.evaluate()
                self.assertTrue(torch.equal(torch.random.get_rng_state(), before))
                self.assertEqual(trainer.model.training, training)
            self.assertEqual(seen_modes, [False, False])

    def test_inherited_identity_score_still_measures_and_saves_expression_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = self.trainer(directory)
            trainer.restore_identity_validation({"identity_validation": {
                "best_score": 0.8, "latest": {"identity_score": 0.8},
            }})
            self.initialize(trainer)
            self.assertEqual(trainer._best_sampled_identity_score, 0.8)
            saved = torch.load(Path(directory) / "latest_checkpoint.pth", weights_only=True)
            calibration = saved["identity_validation"]["calibration"]
            self.assertEqual(calibration["expression_baseline"], 0.4)
            self.assertAlmostEqual(calibration["max_expression_error"], 0.44)
            self.assertEqual(calibration["baseline_progress"], 0.8)
            self.assertAlmostEqual(calibration["min_expression_progress"], 0.72)
            self.assertEqual(calibration["checks_without_improvement"], 0)
            self.assertEqual(calibration["patience"], 5)
            self.assertFalse(getattr(trainer, "stop_training", False))

    def test_higher_identity_cannot_promote_worse_or_copied_expression(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = self.trainer(directory)
            self.initialize(trainer)
            for epoch, expression in enumerate((0.45, 1.0)):
                trainer.epoch = epoch
                self.evaluate(trainer, score=0.85, expression=expression)
                self.assertEqual(trainer._best_sampled_identity_score, 0.8)
            trainer.epoch = 2
            self.evaluate(trainer, score=0.85, expression=0.43)
            self.assertEqual(trainer._best_sampled_identity_score, 0.85)
            saved = torch.load(Path(directory) / "best_identity_checkpoint.pth", weights_only=True)
            calibration = saved["identity_validation"]["calibration"]
            self.assertEqual(calibration["checks_without_improvement"], 0)
            self.assertEqual(calibration["expression_baseline"], 0.4)
            self.assertAlmostEqual(calibration["max_expression_error"], 0.44)

    def test_expression_limit_uses_baseline_without_cap_and_perfect_baseline_allows_a_match(self):
        for expression, expected_limit in ((2.0, 2.2), (0.0, 0.0)):
            with self.subTest(expression=expression), tempfile.TemporaryDirectory() as directory:
                trainer = self.trainer(directory)
                self.initialize(trainer, expression=expression)
                saved = torch.load(Path(directory) / "latest_checkpoint.pth", weights_only=True)
                calibration = saved["identity_validation"]["calibration"]
                self.assertEqual(calibration["max_expression_error"], expected_limit)
                trainer.epoch = 0
                self.evaluate(trainer, score=0.85, expression=expression)
                self.assertEqual(trainer._best_sampled_identity_score, 0.85)
                trainer.epoch = 1
                self.evaluate(trainer, score=0.9, expression=1.0, progress=0.0, direction_hits=0)
                self.assertEqual(trainer._best_sampled_identity_score, 0.85)

    def test_baseline_expression_error_and_progress_cannot_regress_more_than_ten_percent(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = self.trainer(directory)
            self.initialize(trainer, expression=2.0, progress=0.8)
            for epoch, expression, progress in ((0, 2.21, 0.8), (1, 2.0, 0.71)):
                trainer.epoch = epoch
                self.evaluate(trainer, score=0.85, expression=expression, progress=progress)
                self.assertEqual(trainer._best_sampled_identity_score, 0.8)
            trainer.epoch = 2
            self.evaluate(trainer, score=0.85, expression=2.1, progress=0.73)
            self.assertEqual(trainer._best_sampled_identity_score, 0.85)

    def test_baseline_requires_all_six_finite_expression_directions(self):
        for progress, direction_hits in ((0.8, 5), (None, 6)):
            with self.subTest(progress=progress, hits=direction_hits), tempfile.TemporaryDirectory() as directory:
                trainer = self.trainer(directory)
                with self.assertRaisesRegex(RuntimeError, "expression baseline"):
                    self.initialize(trainer, progress=progress, direction_hits=direction_hits)

    def test_default_patience_counts_five_epochs_and_excludes_initial_measurement(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = self.trainer(directory)
            self.initialize(trainer)
            for epoch in range(4):
                trainer.epoch = epoch
                self.evaluate(trainer, score=0.79)
                self.evaluate(trainer, score=0.79)
                self.assertFalse(getattr(trainer, "stop_training", False))
            trainer.epoch = 4
            self.evaluate(trainer, score=0.79)
            self.assertTrue(trainer.stop_training)
            saved = torch.load(Path(directory) / "latest_checkpoint.pth", weights_only=True)
            self.assertEqual(saved["identity_validation"]["calibration"]["checks_without_improvement"], 5)

    def test_identity_promotion_resets_calibration_patience(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = self.trainer(directory, identity_calibration_patience=3)
            self.initialize(trainer)
            for epoch in range(2):
                trainer.epoch = epoch
                self.evaluate(trainer, score=0.79)
            trainer.epoch = 2
            self.evaluate(trainer, score=0.81)
            for epoch in (3, 4):
                trainer.epoch = epoch
                self.evaluate(trainer, score=0.80)
                self.assertFalse(getattr(trainer, "stop_training", False))
            trainer.epoch = 5
            self.evaluate(trainer, score=0.80)
            self.assertTrue(trainer.stop_training)

    def test_resume_restores_patience_and_original_expression_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = self.trainer(directory, identity_calibration_patience=2)
            self.initialize(trainer)
            trainer.epoch = 0
            self.evaluate(trainer, score=0.79, expression=0.42)
            restored = self.trainer(directory, identity_calibration_patience=2)
            restored.load_checkpoint(Path(directory) / "latest_checkpoint.pth")
            with patch("utils.identity_validation.evaluate_sampled_identity",
                       side_effect=AssertionError("Restored baseline must not be remeasured")):
                restored.initialize_identity_validation()
            restored.epoch = 1
            self.evaluate(restored, score=0.79, expression=0.42)
            self.assertTrue(restored.stop_training)
            saved = torch.load(Path(directory) / "latest_checkpoint.pth", weights_only=True)
            calibration = saved["identity_validation"]["calibration"]
            self.assertEqual(calibration["expression_baseline"], 0.4)
            self.assertAlmostEqual(calibration["max_expression_error"], 0.44)
            self.assertEqual(calibration["baseline_progress"], 0.8)
            self.assertAlmostEqual(calibration["min_expression_progress"], 0.72)
            self.assertEqual(calibration["checks_without_improvement"], 2)

    def test_legacy_validation_does_not_enable_expression_gate_or_patience(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = self.trainer(directory, identity_gain_only=False)
            with patch("utils.identity_validation.evaluate_sampled_identity",
                       return_value=metrics(expression=1.0)):
                trainer.validate_sampled_identity()
            self.assertEqual(trainer._best_sampled_identity_score, 0.8)
            saved = torch.load(Path(directory) / "best_identity_checkpoint.pth", weights_only=True)
            self.assertNotIn("calibration", saved["identity_validation"])
            self.assertFalse(getattr(trainer, "stop_training", False))

    def test_unmeasured_identity_reference_can_promote_and_log_a_finite_score(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = self.trainer(directory, identity_gain_only=False)
            trainer._best_sampled_identity_score = None
            with self.assertLogs(trainer.logger, level="INFO"):
                with patch("utils.identity_validation.evaluate_sampled_identity", return_value=metrics()):
                    trainer.validate_sampled_identity()
            self.assertEqual(trainer._best_sampled_identity_score, 0.8)


if __name__ == "__main__":
    unittest.main()
