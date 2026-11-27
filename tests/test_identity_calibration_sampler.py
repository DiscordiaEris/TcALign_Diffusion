import unittest
from types import SimpleNamespace

from scripts.train_stage2 import create_identity_calibration_loader_inputs


class CalibrationSamplerTest(unittest.TestCase):
    def test_frontal_training_pool_balances_edits_and_limits_full_rollouts(self):
        data = SimpleNamespace(metadata=[
            ("fei-001-11.jpg", "left.jpg", "Turn the head toward the left side of the image."),
            ("fei-002-11.jpg", "left.jpg", "Turn the head toward the left side of the image."),
            ("fei-001-11.jpg", "smile.jpg", "Make the person smile."),
            ("fei-001-03.jpg", "right.jpg", "Turn the head toward the right side of the image."),
            ("fei-001-11.jpg", "dark.jpg", "Make the lighting darker."),
        ])
        config = SimpleNamespace(identity_rollout_steps_per_epoch=32, batch_size=1, seed=42)
        selected, sampler = create_identity_calibration_loader_inputs(data, config)
        self.assertEqual(selected.indices, [0, 1, 2])
        self.assertEqual(len(sampler), 32)
        self.assertEqual(sampler.weights.tolist(), [.5, .5, 1.])
        self.assertEqual(len(list(sampler)), 32)

    def test_missing_training_pool_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "frontal"):
            create_identity_calibration_loader_inputs(SimpleNamespace(metadata=[]),
                SimpleNamespace(identity_rollout_steps_per_epoch=32, batch_size=1, seed=42))
