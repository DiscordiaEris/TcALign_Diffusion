import logging
import unittest
from types import SimpleNamespace

import torch

from trainers.stage2_trainer import Stage2Trainer


class RandomLossModel(torch.nn.Module):
    def forward(self, images, texts, **kwargs):
        return {"loss": torch.rand(()), "loss_identity": torch.rand(())}


class ValidationTest(unittest.TestCase):
    def test_exposure_validation_weights_only_requested_lighting_samples(self):
        class ExposureModel(torch.nn.Module):
            def forward(self, images, texts, **kwargs):
                if texts[0] == "pose":
                    return {"loss": torch.tensor(0.)}
                count, value = (1, 4.) if len(texts) == 2 else (3, 2.)
                return {"loss": torch.tensor(value),
                        "loss_lighting_exposure": torch.tensor(value),
                        "lighting_exposure_sample_count": count}

        trainer = Stage2Trainer.__new__(Stage2Trainer)
        trainer.model = ExposureModel()
        trainer.config = SimpleNamespace(seed=42, fid_enabled=False)
        trainer.device = "cpu"
        trainer.logger = logging.getLogger("exposure_validation_test")
        trainer.metrics_computer = None
        trainer.generate_examples = lambda: None
        trainer.val_dataloader = [{
            "image": torch.zeros(len(texts), 3, 2, 2), "text": texts,
            "rendered_image": torch.zeros(len(texts), 3, 2, 2),
            "source_alignment": torch.zeros(len(texts), 2, 3),
            "target_alignment": torch.zeros(len(texts), 2, 3),
        } for texts in (["pose"], ["lighting", "pose"], ["lighting"] * 3)]
        with self.assertLogs(trainer.logger, level="INFO") as captured:
            trainer.evaluate()
        exposure_records = [record for record in captured.records
                            if record.args and record.args[0] == "loss_lighting_exposure"]
        self.assertEqual(len(exposure_records), 1)
        self.assertEqual(exposure_records[0].args[1:3], (2.5, 4))

    def test_validation_reuses_noise_and_preserves_training_rng(self):
        trainer = Stage2Trainer.__new__(Stage2Trainer)
        trainer.model = RandomLossModel()
        trainer.config = SimpleNamespace(seed=42, fid_enabled=False)
        trainer.device = "cpu"
        trainer.logger = logging.getLogger("validation_test")
        trainer.metrics_computer = None
        trainer.generate_examples = lambda: None
        trainer.val_dataloader = [{
            "image": torch.zeros(1, 3, 2, 2), "text": ["smile"],
            "rendered_image": torch.zeros(1, 3, 2, 2),
            "source_alignment": torch.zeros(1, 2, 3),
            "target_alignment": torch.zeros(1, 2, 3),
        } for _ in range(3)]
        torch.manual_seed(123)
        before = torch.get_rng_state().clone()
        first = trainer.evaluate()
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        torch.rand(100)
        second = trainer.evaluate()
        self.assertEqual(first, second)
        self.assertTrue(trainer.model.training)


if __name__ == "__main__":
    unittest.main()
