"""Exercise the real DDIM loop without loading pretrained models or CUDA."""

import unittest
from types import SimpleNamespace

import torch
from diffusers import DDIMScheduler
from torch import nn

from models.diffusion_unet_wrapper import DiffusionUNetWrapper


class ToyIdentityDenoiser(DiffusionUNetWrapper):
    def __init__(self):
        nn.Module.__init__(self)
        self.config = SimpleNamespace(use_bf16=False)
        self.inference_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            steps_offset=1,
            set_alpha_to_one=False,
        )
        self.frozen_prior = nn.Parameter(torch.tensor(0.017), requires_grad=False)
        self.identity_gate = nn.Parameter(torch.tensor(0.0))
        self.calls = []

    def encode_prompt(self, texts):
        return torch.zeros(len(texts), 2, 3)

    def geometry_features(self, geometry, latent_size):
        return None if geometry is None else (geometry * 0.3, geometry * 0.7)

    def forward(self, noisy_latents, timesteps, condition_embeddings,
                source_images, source_latents, *, geometry_residuals=None,
                identity_embeds=None):
        self.calls.append((int(timesteps[0]), condition_embeddings.detach().clone(),
                           None if identity_embeds is None else identity_embeds.detach().clone()))
        time = timesteps[:, None, None, None].float() / 1000
        tokens = condition_embeddings.mean(dim=(1, 2))[:, None, None, None]
        prediction = noisy_latents * (self.frozen_prior + time * 0.02)
        prediction = prediction + tokens * 0.03 + source_images.mean() * 0.01
        prediction = prediction + source_latents * 0.01
        for residual in geometry_residuals or ():
            prediction = prediction + residual * (time + 0.02) * 0.01
        if identity_embeds is not None:
            identity = identity_embeds.mean(dim=(1, 2))[:, None, None, None]
            gain = 0.5 + self.identity_gate.sigmoid()
            # Depend on timestep and latent so a late-bound checkpoint closure
            # changes the derivative even when forward values look correct.
            prediction = prediction + gain * identity * (time + 0.1) * noisy_latents.tanh() * 0.1
        return prediction


class IdentitySamplingGradientTest(unittest.TestCase):
    def inputs(self, *, identity=True, geometry=True):
        return dict(
            condition_embeddings=torch.full((1, 2, 3), 0.4),
            source_images=torch.full((1, 3, 4, 4), 0.2),
            source_latents=torch.linspace(-0.3, 0.5, 16).reshape(1, 4, 2, 2),
            guidance_scale=1.5,
            num_inference_steps=50,
            strength=0.65,
            geometry=torch.full((1, 4, 2, 2), 0.6) if geometry else None,
            identity_embeds=torch.full((1, 1, 3), 0.7) if identity else None,
        )

    def generator(self):
        return torch.Generator(device="cpu").manual_seed(19)

    def test_inference_and_training_use_the_same_complete_ddim_schedule(self):
        # Removing a DDIM step, changing CFG negatives, or sharing scheduler
        # state between calls breaks the public inference/training contract.
        model = ToyIdentityDenoiser()
        scheduler_timesteps = model.inference_scheduler.timesteps.clone()
        inference = model.sample(**self.inputs(), generator=self.generator())
        calls = list(model.calls)
        model.calls.clear()
        differentiable = model.sample_train(**self.inputs(), generator=self.generator())
        torch.testing.assert_close(inference, differentiable, rtol=0, atol=0)
        self.assertFalse(inference.requires_grad)
        self.assertTrue(differentiable.requires_grad)
        self.assertEqual(inference.dtype, torch.float32)
        expected_steps = [step for step in range(621, 0, -20) for _ in range(2)]
        self.assertEqual([step for step, _, _ in calls], expected_steps)
        self.assertEqual([step for step, _, _ in model.calls], expected_steps)
        for conditional, unconditional in zip(calls[::2], calls[1::2]):
            torch.testing.assert_close(conditional[1], torch.full((1, 2, 3), 0.4))
            torch.testing.assert_close(conditional[2], torch.full((1, 1, 3), 0.7))
            self.assertEqual(int(unconditional[1].count_nonzero()), 0)
            self.assertEqual(int(unconditional[2].count_nonzero()), 0)
        torch.testing.assert_close(model.inference_scheduler.timesteps, scheduler_timesteps)
        repeated = model.sample(**self.inputs(), generator=self.generator())
        torch.testing.assert_close(inference, repeated, rtol=0, atol=0)

    def test_checkpoint_recomputation_preserves_timestep_dependent_gate_gradient(self):
        # Capturing the loop's final timestep/tokens/ID in a closure corrupts
        # backward; compare with the same complete loop without checkpointing.
        checkpointed, direct = ToyIdentityDenoiser(), ToyIdentityDenoiser()
        actual = checkpointed.sample_train(**self.inputs(), generator=self.generator())
        expected = direct.sample_train(**self.inputs(), generator=self.generator(),
                                       checkpoint_denoiser=False)
        actual.square().mean().backward()
        expected.square().mean().backward()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertIsNotNone(checkpointed.identity_gate.grad)
        self.assertGreater(float(checkpointed.identity_gate.grad.abs()), 1e-7)
        self.assertTrue(bool(torch.isfinite(checkpointed.identity_gate.grad)))
        torch.testing.assert_close(checkpointed.identity_gate.grad, direct.identity_gate.grad,
                                   rtol=1e-6, atol=1e-7)
        self.assertIsNone(checkpointed.frozen_prior.grad)
        self.assertIsNone(direct.frozen_prior.grad)
        self.assertEqual(len(direct.calls), 64)
        self.assertGreater(len(checkpointed.calls), len(direct.calls))

    def test_training_can_sample_without_optional_identity_or_geometry(self):
        model = ToyIdentityDenoiser()
        inputs = self.inputs(identity=False, geometry=False)
        inputs["guidance_scale"] = 1.0
        inference = model.sample(**inputs, generator=self.generator())
        training = model.sample_train(**inputs, generator=self.generator())
        torch.testing.assert_close(inference, training, rtol=0, atol=0)
        self.assertEqual([step for step, _, _ in model.calls],
                         list(range(621, 0, -20)) * 2)

    def test_outer_training_amp_does_not_change_cfg_negative_prompt_encoding(self):
        model = ToyIdentityDenoiser()
        projection = nn.Linear(3, 3).requires_grad_(False)
        prompt_autocast = []
        def encode(texts):
            prompt_autocast.append(torch.is_autocast_enabled("cpu"))
            return projection(torch.ones(len(texts), 2, 3))
        model.encode_prompt = encode
        expected = model.sample(**self.inputs(), generator=self.generator())
        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual = model.sample_train(**self.inputs(), generator=self.generator())
        self.assertEqual(prompt_autocast, [False, False])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
