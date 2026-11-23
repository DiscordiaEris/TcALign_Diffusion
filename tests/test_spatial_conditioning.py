import unittest

import torch

from models.geometry_adapter import GeometryAdapter, align_geometry


class GeometryAdapterTest(unittest.TestCase):
    def test_canonical_face_is_warped_into_photo_coordinates(self):
        geometry = torch.zeros(1, 7, 32, 32)
        geometry[:, :3] = -1
        geometry[:, 6, 8:24, 8:24] = 1
        alignment = torch.tensor([[[0.5, 0., 0.25], [0., 0.5, 0.0]]])
        warped = align_geometry(geometry, alignment, (32, 32))
        points = (warped[0, 6] > 0.5).nonzero().float()
        self.assertEqual(len(points), 64)
        self.assertAlmostEqual(points[:, 1].mean().item(), 19.5)
        self.assertTrue(torch.equal(warped[:, :3], geometry[:, :3]))

    def test_residuals_match_denoiser_levels_and_start_at_zero(self):
        adapter = GeometryAdapter((16, 32, 64, 64))
        residuals = adapter(torch.randn(2, 7, 80, 80), (16, 16))
        self.assertEqual([tuple(t.shape) for t in residuals],
                         [(2, 16, 16, 16), (2, 32, 8, 8), (2, 64, 4, 4), (2, 64, 2, 2)])
        self.assertTrue(all(torch.count_nonzero(t) == 0 for t in residuals))
        sum(t.mean() for t in residuals).backward()
        self.assertTrue(all(layer.bias.grad.abs().sum() > 0 for layer in adapter.projections))

    def test_geometry_changes_the_residual_after_projection_learns(self):
        adapter = GeometryAdapter((16,))
        with torch.no_grad():
            adapter.projections[0].weight.fill_(0.01)
        face = torch.randn(1, 7, 32, 32)
        first, = adapter(face, (16, 16))
        second, = adapter(face.flip(-1), (16, 16))
        self.assertFalse(torch.allclose(first, second))


if __name__ == "__main__":
    unittest.main()
