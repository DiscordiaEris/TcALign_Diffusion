import logging
import unittest
from types import SimpleNamespace

import torch

from trainers.stage2_trainer import Stage2Trainer


class RandomLossModel(torch.nn.Module):
    def forward(self, images, texts, **kwargs):
        return {"loss": torch.rand(()), "loss_identity": torch.rand(())}


class ValidationTest(unittest.TestCase):
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
