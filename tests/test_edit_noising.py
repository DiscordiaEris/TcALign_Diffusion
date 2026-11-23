import unittest

import torch
from diffusers import DDPMScheduler

from utils import losses


class EditNoisingTest(unittest.TestCase):
    def test_same_source_noise_requires_different_targets_for_different_edits(self):
        scheduler = DDPMScheduler(num_train_timesteps=1000, clip_sample=False)
        source = torch.zeros(2, 4, 4, 4)
        target = torch.stack([torch.ones(4, 4, 4), -torch.ones(4, 4, 4)])
        noise = torch.ones_like(source) * 0.25
        ts = torch.tensor([600, 600])
        pair = getattr(losses, "make_edit_training_pair", None)
        self.assertIsNotNone(pair, "Source-start editing needs an explicit transport target")
        noisy, epsilon = pair(scheduler, target, source, noise, ts, torch.ones(2, dtype=torch.bool))
        self.assertTrue(torch.equal(noisy[0], noisy[1]))
        self.assertFalse(torch.equal(epsilon[0], epsilon[1]))
        alpha = scheduler.alphas_cumprod[ts, None, None, None]
        recovered = (noisy - (1 - alpha).sqrt() * epsilon) / alpha.sqrt()
        torch.testing.assert_close(recovered, target)

    def test_standard_branch_keeps_the_original_gaussian_target(self):
        scheduler = DDPMScheduler(num_train_timesteps=1000)
        source, target, noise = [torch.randn(2, 4, 4, 4) for _ in range(3)]
        ts = torch.tensor([200, 700])
        noisy, epsilon = losses.make_edit_training_pair(
            scheduler, target, source, noise, ts, torch.zeros(2, dtype=torch.bool))
        torch.testing.assert_close(noisy, scheduler.add_noise(target, noise, ts))
        torch.testing.assert_close(epsilon, noise)

    def test_velocity_conversion_recovers_target_in_source_branch(self):
        scheduler = DDPMScheduler(num_train_timesteps=1000, prediction_type="v_prediction")
        source, target, noise = [torch.randn(2, 4, 4, 4) for _ in range(3)]
        ts = torch.tensor([350, 650])
        noisy, epsilon = losses.make_edit_training_pair(
            scheduler, target, source, noise, ts, torch.ones(2, dtype=torch.bool))
        alpha = scheduler.alphas_cumprod[ts, None, None, None]
        velocity = alpha.sqrt() * epsilon - (1 - alpha).sqrt() * target
        torch.testing.assert_close(alpha.sqrt() * noisy - (1 - alpha).sqrt() * velocity, target)


if __name__ == "__main__":
    unittest.main()
