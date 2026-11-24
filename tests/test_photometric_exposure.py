import io
import math
import unittest

import torch

from models.photometric_exposure import (
    PhotometricExposureHead,
    aligned_face_luminance,
    apply_photometric_exposure,
)


class PhotometricExposureTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.head = PhotometricExposureHead()
        self.tokens = torch.randn(2, 77, 768)
        self.alignment = torch.eye(2, 3)[None].expand(2, -1, -1)

    def test_new_head_predicts_source_exposure_without_a_text_rule(self):
        source_log = torch.tensor([math.log(.2), math.log(.7)])
        prediction = self.head(self.tokens, source_log)
        self.assertEqual(tuple(prediction.shape), (2,))
        self.assertTrue(torch.equal(prediction, source_log))
        self.assertTrue(torch.equal(self.head(-self.tokens, source_log), source_log))

    def test_luminance_uses_the_aligned_face_instead_of_background(self):
        rgb = torch.full((1, 3, 16, 16), .8)
        rgb[..., :10] = .2
        alignment = torch.tensor([[[.25, 0., -.5], [0., .5, 0.]]])
        measured = aligned_face_luminance(rgb * 2 - 1, alignment)
        self.assertAlmostEqual(float(measured[0]), .2, places=6)
        # A red image checks the RGB luminance weights independently.
        red = torch.full((1, 3, 8, 8), -1.)
        red[:, 0] = 1
        measured_red = aligned_face_luminance(red, self.alignment[:1])
        self.assertAlmostEqual(float(measured_red[0]), .2126, places=6)

    def test_dimming_scales_existing_texture_without_blurring_it(self):
        rgb = torch.full((2, 3, 16, 16), .25)
        rgb[..., 1::2] = .75
        source = torch.full_like(rgb, -.5)  # source RGB luminance .25
        corrected = apply_photometric_exposure(
            self.head, source, rgb * 2 - 1, self.tokens, self.alignment,
            torch.tensor([True, False]),
        )
        # Generated luminance .5 becomes .25 with a uniform factor .5.
        self.assertTrue(torch.allclose((corrected[0] + 1) / 2, rgb[0] * .5, atol=1e-6))
        self.assertTrue(torch.equal(corrected[1], (rgb * 2 - 1)[1]))

    def test_prediction_is_absolute_and_independent_of_generated_brightness(self):
        source = torch.full((2, 3, 16, 16), -.6)  # source RGB .2
        generated = torch.empty_like(source)
        generated[0] = .6  # RGB .8
        generated[1] = -.2  # RGB .4
        corrected = apply_photometric_exposure(
            self.head, source, generated, self.tokens, self.alignment,
            torch.ones(2, dtype=torch.bool),
        )
        self.assertTrue(torch.allclose(corrected, source, atol=1e-6))

    def test_disabled_and_unrequested_batches_are_exact_identity(self):
        generated = torch.randn(2, 3, 8, 8).half()
        source = torch.zeros_like(generated)
        # Invalid text values must never be evaluated on a bypassed batch.
        tokens = torch.full_like(self.tokens, float("nan"))
        disabled = apply_photometric_exposure(
            self.head, source, generated, tokens, self.alignment,
            torch.ones(2, dtype=torch.bool), enabled=False,
        )
        unrequested = apply_photometric_exposure(
            self.head, source, generated, tokens, self.alignment,
            torch.zeros(2, dtype=torch.bool),
        )
        self.assertIs(disabled, generated)
        self.assertIs(unrequested, generated)

    def test_unrequested_mixed_rows_preserve_dtype_and_values_without_clamping(self):
        source = torch.full((2, 3, 8, 8), -.5, dtype=torch.float16)
        generated = torch.full_like(source, .5)
        generated[1, 0, 0, 0] = 1.25
        tokens = self.tokens.clone()
        tokens[1] = float("nan")
        corrected = apply_photometric_exposure(
            self.head, source, generated, tokens, self.alignment,
            torch.tensor([True, False]),
        )
        self.assertEqual(corrected.dtype, generated.dtype)
        self.assertTrue(torch.equal(corrected[1], generated[1]))
        self.assertTrue(torch.isfinite(corrected[0]).all())

    def test_brightness_saturation_and_maximum_gain_are_explicit_limits(self):
        source = torch.full((2, 3, 16, 16), .6)  # target RGB .8
        rgb = torch.full_like(source, .1)
        rgb[0, ..., 1::2] = .9  # mean .5, so requested gain is 1.6
        rgb[1] = .01  # requested gain 80, limited to 2
        corrected = apply_photometric_exposure(
            self.head, source, rgb * 2 - 1, self.tokens, self.alignment,
            torch.ones(2, dtype=torch.bool), max_gain=2.,
        )
        result = (corrected + 1) / 2
        self.assertTrue(torch.allclose(result[0], (rgb[0] * 1.6).clamp(0, 1), atol=1e-6))
        self.assertTrue(torch.allclose(result[1], torch.full_like(result[1], .02), atol=1e-6))
        self.assertLess(float(result[0].mean()), .8)
        self.assertGreaterEqual(float(result.min()), 0)
        self.assertLessEqual(float(result.max()), 1)

    def test_black_images_and_their_gradients_remain_finite(self):
        source = torch.full((2, 3, 8, 8), -1.)
        source[0] = .2
        generated = torch.full_like(source, -1.)
        generated[1] = .8
        generated.requires_grad_()
        corrected = apply_photometric_exposure(
            self.head, source, generated, self.tokens, self.alignment,
            torch.ones(2, dtype=torch.bool),
        )
        corrected.square().mean().backward()
        self.assertTrue(torch.isfinite(corrected).all())
        self.assertTrue(torch.equal(corrected[0], generated[0]))
        self.assertTrue(torch.isfinite(generated.grad).all())
        for parameter in self.head.parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_final_rgb_supervision_reaches_the_learnable_head(self):
        source = torch.full((2, 3, 8, 8), -.2)
        generated = torch.full_like(source, .6, requires_grad=True)
        corrected = apply_photometric_exposure(
            self.head, source, generated, self.tokens, self.alignment,
            torch.tensor([True, False]),
        )
        corrected[0].square().mean().backward()
        gradient = sum(float(p.grad.abs().sum()) for p in self.head.parameters() if p.grad is not None)
        self.assertGreater(gradient, 0)
        self.assertTrue(torch.isfinite(generated.grad).all())
        self.assertTrue(torch.equal(generated.grad[1], torch.zeros_like(generated.grad[1])))

    def test_weight_reload_preserves_predictions_and_text_conditioning(self):
        with torch.no_grad():
            for parameter in self.head.parameters():
                parameter.normal_(0, .03)
        source_log = torch.tensor([math.log(.3), math.log(.6)])
        expected = self.head(self.tokens, source_log)
        self.assertFalse(torch.allclose(expected, self.head(-self.tokens, source_log)))
        memory = io.BytesIO()
        torch.save(self.head.state_dict(), memory)
        memory.seek(0)
        restored = PhotometricExposureHead()
        restored.load_state_dict(torch.load(memory, weights_only=True))
        self.assertTrue(torch.equal(restored(self.tokens, source_log), expected))

    def test_mask_must_be_an_explicit_per_row_boolean(self):
        images = torch.zeros(2, 3, 8, 8)
        for mask in (torch.ones(2), torch.ones(2, 1, dtype=torch.bool)):
            with self.subTest(mask=mask):
                with self.assertRaises((TypeError, ValueError)):
                    apply_photometric_exposure(
                        self.head, images, images, self.tokens, self.alignment, mask,
                    )


if __name__ == "__main__":
    unittest.main()
