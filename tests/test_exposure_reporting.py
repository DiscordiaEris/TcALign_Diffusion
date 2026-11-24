import unittest
from types import SimpleNamespace

import torch

from models.tcalign_diffusion import TcAlignDiffusion


class ExposureReportingTest(unittest.TestCase):
    def forward(self, texts, max_samples):
        class ImageDiffusion:
            geometry_adapter = None
            noise_scheduler = SimpleNamespace(
                config=SimpleNamespace(prediction_type="epsilon"),
                alphas_cumprod=torch.full((100,), 0.5),
                add_noise=lambda image, noise, timesteps: (image + noise) / 2 ** 0.5,
            )

            def encode_image(self, images, deterministic=False):
                return images

            def encode_prompt(self, texts):
                return torch.zeros(len(texts), 1, 1)

            def geometry_features(self, geometry, latent_size):
                return None

            def __call__(self, noisy, *args, **kwargs):
                return torch.zeros_like(noisy)

            def decode_latent_train(self, latents):
                return latents

        model = TcAlignDiffusion.__new__(TcAlignDiffusion)
        torch.nn.Module.__init__(model)
        model.config = SimpleNamespace(
            lambda_3d=0., lambda_id=0., lambda_control=0., lambda_lighting_exposure=1.,
            lambda_clip=0., lambda_diffusion=1., lambda_diffusion_edit=0.,
            cfg_dropout_prob=0., num_train_timesteps=100, source_edit_probability=0.,
            auxiliary_max_timestep=100, max_clip_grad_samples=max_samples,
        )
        model.diffusion_unet = ImageDiffusion()
        model.identity_loss = None
        model._init_diffusion_modules = lambda: None
        model.initialize_identity_loss = lambda: None
        model.predict_delta = lambda theta, texts, rendered: torch.zeros_like(theta)
        model.text_3d_editor = SimpleNamespace(parse_attributes=lambda text: [text])
        model.diffusion_mapper = lambda delta, latents, text: text
        images = torch.zeros(len(texts), 3, 2, 2)
        theta = torch.zeros(len(texts), 254)
        alignment = torch.eye(2, 3).expand(len(texts), -1, -1)
        return model(images, texts, theta_0=theta, target_images=images - 0.2,
                     target_theta=theta, source_alignment=alignment,
                     target_alignment=alignment)

    def test_nonlighting_decoded_images_do_not_report_exposure_observations(self):
        result = self.forward(["pose", "lighting"], max_samples=1)
        self.assertNotIn("loss_lighting_exposure", result)
        self.assertNotIn("lighting_exposure_sample_count", result)

    def test_exposure_reports_number_of_lighting_images_in_decoded_subset(self):
        result = self.forward(["pose", "lighting", "lighting"], max_samples=3)
        self.assertEqual(result["lighting_exposure_sample_count"], 2)
        self.assertTrue(torch.isfinite(result["loss_lighting_exposure"]))


if __name__ == "__main__":
    unittest.main()
