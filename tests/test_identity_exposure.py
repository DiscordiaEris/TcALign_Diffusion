import unittest

import torch
from torch import nn
from torch.nn import functional as F

from models.face_identity_loss import FaceIdentityLoss


class IdentityExposureTest(unittest.TestCase):
    def make_loss(self, normalize):
        loss = FaceIdentityLoss.__new__(FaceIdentityLoss)
        nn.Module.__init__(loss)
        loss.normalize_exposure = normalize
        loss.encoder = nn.Sequential(nn.Flatten(), nn.Linear(3 * 160 * 160, 16, bias=False))
        loss.encoder.requires_grad_(False)
        return loss

    def test_global_dimming_is_not_identity_change(self):
        torch.manual_seed(3)
        source = torch.rand(1, 3, 32, 32) * .4 + .2
        dimmed = source * .25
        source, dimmed = source * 2 - 1, dimmed * 2 - 1
        alignment = torch.eye(2, 3)[None]
        loss = self.make_loss(True)
        value = loss(dimmed, source, dimmed, alignment, alignment)
        self.assertLess(float(value), 1e-5)
        # The public default embedding remains the raw-exposure monitoring metric.
        raw_similarity = F.cosine_similarity(loss.embed(source, alignment), loss.embed(dimmed, alignment))
        self.assertLess(float(raw_similarity), .95)

    def test_identity_structure_still_receives_finite_gradient(self):
        torch.manual_seed(7)
        source = torch.rand(1, 3, 32, 32) * .5 - .5
        generated = (source + torch.randn_like(source) * .15).requires_grad_()
        alignment = torch.eye(2, 3)[None]
        loss = self.make_loss(True)
        value = loss(generated, source, source, alignment, alignment)
        value.backward()
        self.assertGreater(float(value), 0)
        self.assertTrue(torch.isfinite(generated.grad).all())
        self.assertGreater(float(generated.grad.abs().sum()), 0)
        self.assertTrue(all(parameter.grad is None for parameter in loss.encoder.parameters()))

    def test_black_crop_is_finite(self):
        loss = self.make_loss(True)
        black = torch.full((1, 3, 16, 16), -1., requires_grad=True)
        alignment = torch.eye(2, 3)[None]
        features = loss.embed(black, alignment, normalize_exposure=True)
        features.sum().backward()
        self.assertTrue(torch.isfinite(features).all())
        self.assertTrue(torch.isfinite(black.grad).all())


if __name__ == "__main__":
    unittest.main()
