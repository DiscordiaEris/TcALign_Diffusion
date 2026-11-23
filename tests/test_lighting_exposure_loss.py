import unittest

import torch

from utils import losses


class LightingExposureLossTest(unittest.TestCase):
    def exposure_loss(self, predicted, source, target, attributes, alignment=None):
        if alignment is None:
            alignment = torch.tensor([[[1., 0., 0.], [0., 1., 0.]]]).expand(len(predicted), -1, -1)
        return losses.requested_lighting_exposure_loss(
            predicted, source, target, alignment, attributes
        )

    def test_copying_source_has_unit_error_and_matching_target_has_zero(self):
        source = torch.full((1, 3, 16, 16), 0.2)
        target = torch.full_like(source, -0.6)
        self.assertAlmostEqual(float(self.exposure_loss(source, source, target, [["lighting"]])),
                               1., places=6)
        self.assertEqual(float(self.exposure_loss(target, source, target, [["lighting"]])), 0.)

    def test_darker_target_pushes_generated_exposure_down(self):
        source = torch.full((1, 3, 16, 16), 0.2)
        target = torch.full_like(source, -0.6)
        predicted = source.clone().requires_grad_()
        self.exposure_loss(predicted, source, target, [["lighting"]]).backward()
        self.assertTrue((predicted.grad > 0).all())
        after = predicted.detach() - predicted.grad * 10
        self.assertLess(float(after.mean()), float(predicted.mean()))
        self.assertLess(float(self.exposure_loss(after, source, target, [["lighting"]])), 1.)

    def test_nonlighting_rows_receive_no_exposure_gradient(self):
        source = torch.full((2, 3, 16, 16), 0.2, requires_grad=True)
        target = torch.full_like(source, -0.6, requires_grad=True)
        predicted = source.detach().clone().requires_grad_()
        value = self.exposure_loss(predicted, source, target, [["lighting"], ["pose"]])
        value.backward()
        self.assertGreater(float(predicted.grad[0].abs().sum()), 0.)
        self.assertEqual(float(predicted.grad[1].abs().sum()), 0.)
        self.assertIsNone(source.grad)
        self.assertIsNone(target.grad)

    def test_without_lighting_returns_differentiable_zero(self):
        source = torch.full((2, 3, 16, 16), 0.2)
        target = torch.full_like(source, -0.6)
        predicted = source.clone().requires_grad_()
        value = self.exposure_loss(predicted, source, target, [["pose"], ["expression"]])
        self.assertEqual(float(value), 0.)
        value.backward()
        self.assertTrue(torch.equal(predicted.grad, torch.zeros_like(predicted)))

    def test_shared_source_alignment_measures_face_region_instead_of_whole_photo(self):
        source = torch.full((1, 3, 16, 16), 0.5)
        source[..., 8:] = -0.5
        target = -source
        # Both photos have mean zero, but the requested face region gets darker.
        alignment = torch.tensor([[[0.4, 0., -0.5], [0., 0.4, 0.]]])
        value = self.exposure_loss(source, source, target, [["lighting"]], alignment)
        self.assertAlmostEqual(float(value), 1., places=6)

    def test_tiny_exposure_displacements_have_bounded_gradients(self):
        source = torch.zeros(1, 3, 16, 16)
        target = torch.full_like(source, 1e-7)
        predicted = source.clone().requires_grad_()
        value = self.exposure_loss(predicted, source, target, [["lighting"]])
        value.backward()
        self.assertTrue(torch.isfinite(value))
        self.assertTrue(torch.isfinite(predicted.grad).all())
        self.assertLess(float(predicted.grad.abs().max()), 1e-5)


if __name__ == "__main__":
    unittest.main()
