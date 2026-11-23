import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from data import transforms
from data.datasets import FFHQDataset


class PairAlignmentTest(unittest.TestCase):
    def register(self, image, source, target):
        register = getattr(transforms, "align_paired_target", None)
        self.assertIsNotNone(register, "Paired targets need registration into the source frame")
        return register(image, source, target)

    def test_patch_moves_to_source_frame_without_mirroring_or_black_border(self):
        image = torch.full((3, 64, 64), -0.5)
        image[0, 28:36, 34:38] = 1
        image[1, 28:36, 42:46] = 1
        source = torch.tensor([[0.5, 0., -0.25], [0., 0.5, 0.]])
        target = torch.tensor([[0.5, 0., 0.25], [0., 0.5, 0.]])
        registered = self.register(image, source, target)
        # The target crop is 16 pixels to the right of the source crop.
        self.assertTrue(torch.allclose(registered[0, 28:36, 18:22], torch.ones(8, 4)))
        self.assertTrue(torch.allclose(registered[1, 28:36, 26:30], torch.ones(8, 4)))
        self.assertTrue(torch.allclose(registered[:, :, -16:], torch.full((3, 64, 16), -0.5)))

    def test_source_aligned_crop_preserves_original_target_canonical_coordinates(self):
        axis = (torch.arange(128) + 0.5) / 64 - 1
        y, x = torch.meshgrid(axis, axis, indexing="ij")
        image = torch.stack((x, y, (x * x + y * y) / 2))
        source = torch.tensor([[0.45, 0., -0.125], [0., 0.45, 0.0625]])
        target = torch.tensor([[0.55, 0., 0.1875], [0., 0.55, -0.125]])
        registered = self.register(image, source, target)

        def crop(portrait, alignment):
            grid = F.affine_grid(alignment[None], (1, 3, 48, 48), align_corners=False)
            return F.grid_sample(portrait[None], grid, align_corners=False)

        self.assertTrue(torch.allclose(crop(registered, source), crop(image, target), atol=2e-4))

    def test_dataset_opt_in_changes_only_target_pixels_and_alignment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images" / "val"
            images.mkdir(parents=True)
            source_pixels = np.full((64, 64, 3), 64, np.uint8)
            target_pixels = np.full((64, 64, 3), 128, np.uint8)
            target_pixels[28:36, 36:44, 0] = 255
            Image.fromarray(source_pixels).save(images / "source.png")
            Image.fromarray(target_pixels).save(images / "target.png")
            with (root / "metadata_val.csv").open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(("source", "target", "text"))
                writer.writerow(("source.png", "target.png", "Turn the head left."))
            cache = root / "deep3d_params" / "aligned_sh_v2_64" / "val"
            cache.mkdir(parents=True)
            provisional = FFHQDataset.__new__(FFHQDataset)
            provisional.image_dir = images
            provisional.cache_dir = cache
            source_alignment = np.array([[0.5, 0., -0.25], [0., 0.5, 0.]], np.float32)
            target_alignment = np.array([[0.5, 0., 0.25], [0., 0.5, 0.]], np.float32)
            for name, alignment, value in (("source.png", source_alignment, 1),
                                           ("target.png", target_alignment, 2)):
                np.savez(provisional._cache_path(name), params=np.full(254, value, np.float32),
                         rendered=np.zeros((64, 64, 3), np.uint8), alignment=alignment)

            def dataset(enabled):
                config = SimpleNamespace(image_size=64, align_paired_targets=enabled)
                return FFHQDataset(root, split="val", config=config, precompute_3d=False,
                                   transform=transforms.create_transforms(64, "val"),
                                   rendered_transform=transforms.create_rendered_transform(64))

            original = dataset(False)[0]
            registered = dataset(True)[0]
            self.assertTrue(torch.equal(registered["image"], original["image"]))
            self.assertTrue(torch.equal(registered["target_params"], original["target_params"]))
            self.assertTrue(torch.equal(registered["params"], original["params"]))
            self.assertTrue(torch.equal(registered["rendered_image"], original["rendered_image"]))
            self.assertTrue(torch.equal(registered["target_alignment"], registered["source_alignment"]))
            self.assertTrue(torch.equal(original["target_alignment"], torch.from_numpy(target_alignment)))
            self.assertTrue(torch.allclose(registered["target_image"][0, 28:36, 20:28], torch.ones(8, 8)))
            with np.load(provisional._cache_path("target.png")) as cached:
                np.testing.assert_array_equal(cached["alignment"], target_alignment)


if __name__ == "__main__":
    unittest.main()
