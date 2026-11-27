import unittest
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from models.tcalign_diffusion import TcAlignDiffusion


class IdentityRolloutObjectiveTest(unittest.TestCase):
    def test_identity_loss_supervises_final_sample_without_target_latent_regression(self):
        model = TcAlignDiffusion.__new__(TcAlignDiffusion)
        nn.Module.__init__(model)
        calls = []
        class Diffusion(nn.Module):
            geometry_adapter = None
            def __init__(self):
                super().__init__()
                self.identity_gain_logit = nn.Parameter(torch.tensor(.2))
                self.frozen_prior = nn.Parameter(torch.tensor(.5), requires_grad=False)
            def encode_image(self, image, **kwargs): return image
            def encode_prompt(self, texts): return torch.zeros(len(texts), 2, 3)
            def sample_train(self, condition, source, latent, **kwargs):
                calls.append(kwargs)
                return self.identity_gain_logit.expand_as(source)
            def decode_latent_train(self, z): return z
        class Identity(nn.Module):
            def forward(self, generated, source, target, *args): return F.mse_loss(generated, source)
        model.diffusion_unet = Diffusion()
        model.identity_loss = Identity()
        model.photometric_head = None
        model._init_diffusion_modules = lambda: None
        model.initialize_identity_loss = lambda: None
        model.predict_delta = lambda theta, texts, rendered: torch.zeros_like(theta)
        model.encode_source_identity = lambda images: None
        model.diffusion_mapper = lambda delta, latent, tokens: tokens
        model.text_3d_editor = SimpleNamespace(parse_attributes=lambda text: ["pose"])
        model.config = SimpleNamespace(guidance_scale=1.5, num_inference_steps=50,
            inference_strength=.65, lambda_id=1., lambda_control=0., lambda_lighting_exposure=0.)
        x = torch.full((1, 3, 8, 8), .6)
        result = model.forward_identity_rollout(x, ["turn left"], theta_0=torch.zeros(1, 254),
            target_images=torch.zeros_like(x), source_rendered=None,
            source_alignment=torch.eye(2, 3)[None], target_alignment=torch.eye(2, 3)[None])
        result["loss"].backward()
        self.assertLess(float(model.diffusion_unet.identity_gain_logit.grad), 0)
        self.assertIsNone(model.diffusion_unet.frozen_prior.grad)
        self.assertIn("loss_final_identity", result)
        self.assertNotIn("loss_diffusion", result)
        self.assertEqual(calls[0]["num_inference_steps"], 50)
        self.assertEqual(calls[0]["guidance_scale"], 1.5)
        self.assertEqual(calls[0]["strength"], .65)


if __name__ == "__main__": unittest.main()
