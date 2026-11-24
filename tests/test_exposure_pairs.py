import csv
import hashlib
import importlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from data.datasets import FFHQDataset
from data.transforms import create_rendered_transform, create_transforms


class ExposurePairsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "original"
        self.output = Path(self.temporary.name) / "derived"
        self.original_rows = {}
        for split, identity in (("train", 1), ("val", 2)):
            directory = self.root / "images" / split
            cache = self.root / "deep3d_params" / "aligned_sh_v2_32" / split
            directory.mkdir(parents=True)
            cache.mkdir(parents=True)
            for shot, level in ((3, 150), (11, 160), (12, 155), (13, 120), (14, 64)):
                name = f"fei-{identity:03d}-{shot:02d}.png"
                pixels = np.full((32, 32, 3), 32, dtype=np.uint8)
                pixels[4:28, 4:28] = level
                Image.fromarray(pixels).save(directory / name)
                params = np.arange(254, dtype=np.float32) / 1000
                params[-27:] = 0.2
                np.savez(
                    self.cache_path(self.root, split, name),
                    params=params,
                    alignment=np.array([[0.5, 0, 0], [0, 0.5, 0]], np.float32),
                    rendered=np.full((8, 8, 3), 100, dtype=np.uint8),
                )
            stem = f"fei-{identity:03d}"
            rows = [
                {
                    "source": f"{stem}-11.png",
                    "target": f"{stem}-12.png",
                    "text": "Make the person smile.",
                    "annotation": "keep expression",
                },
                {
                    "source": f"{stem}-11.png",
                    "target": f"{stem}-03.png",
                    "text": "Turn the head toward the left side of the image.",
                    "annotation": "keep pose",
                },
                {
                    "source": f"{stem}-11.png",
                    "target": f"{stem}-13.png",
                    "text": "Dim the lighting on the face.",
                    "annotation": "bad old dim",
                },
                {
                    "source": f"{stem}-11.png",
                    "target": f"{stem}-14.png",
                    "text": "Make the lighting on the face very dim.",
                    "annotation": "bad old very dim",
                },
                {
                    "source": f"{stem}-13.png",
                    "target": f"{stem}-11.png",
                    "text": "Use bright, even lighting on the face.",
                    "annotation": "old bright",
                },
            ]
            self.original_rows[split] = rows
            with (self.root / f"metadata_{split}.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

    def cache_path(self, root, split, name):
        # Exercise the consuming dataset's cache key instead of the producer's helper.
        dataset = FFHQDataset.__new__(FFHQDataset)
        dataset.image_dir = root / "images" / split
        dataset.cache_dir = root / "deep3d_params" / "aligned_sh_v2_32" / split
        return dataset._cache_path(name)

    def prepare(self):
        spec = importlib.util.find_spec("scripts.prepare_exposure_pairs")
        self.assertIsNotNone(
            spec, "Controlled exposure pairs need a preparation command"
        )
        module = importlib.import_module("scripts.prepare_exposure_pairs")
        return module.prepare(self.root, self.output, image_size=32)

    def rows(self, split):
        with (self.output / f"metadata_{split}.csv").open(newline="") as handle:
            return list(csv.DictReader(handle))

    def snapshot(self):
        return {
            str(path.relative_to(self.root)): (
                path.stat().st_mtime_ns,
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )
            for path in self.root.rglob("*")
            if path.is_file()
        }

    def test_replaces_wrong_lighting_pairs_and_preserves_other_metadata_and_splits(
        self,
    ):
        self.prepare()
        for split, identity in (("train", "001"), ("val", "002")):
            rows = self.rows(split)
            self.assertEqual(rows[:2], self.original_rows[split][:2])
            self.assertEqual(len(rows), 10)
            synthetic = rows[2:]
            self.assertTrue(
                all(row["source"].startswith(f"fei-{identity}-") for row in synthetic)
            )
            by_shot = {
                shot: [row for row in synthetic if f"-{shot}.png" in row["source"]]
                for shot in ("11", "13", "14")
            }
            self.assertEqual(
                [len(by_shot[shot]) for shot in ("11", "13", "14")], [2, 3, 3]
            )
            self.assertFalse(any("bright" in row["text"] for row in by_shot["11"]))
            self.assertEqual(
                {row["text"] for row in synthetic},
                {
                    "Use bright, even lighting on the face.",
                    "Use dim lighting on the face.",
                    "Use very dim lighting on the face.",
                },
            )
            self.assertTrue(all(row["target"].endswith(".png") for row in synthetic))

    def test_uses_aligned_normal_reference_and_produces_ordered_lossless_exposures(
        self,
    ):
        self.prepare()
        rows = self.rows("train")[2:]
        for shot in ("11", "13", "14"):
            selected = [row for row in rows if row["source"] == f"fei-001-{shot}.png"]
            values = {}
            for row in selected:
                with Image.open(self.output / "images/train" / row["target"]) as image:
                    self.assertEqual(image.format, "PNG")
                    self.assertEqual(image.size, (32, 32))
                    center = np.array(image)[10:22, 10:22]
                values[row["text"]] = float(center.mean())
            self.assertAlmostEqual(
                values["Use dim lighting on the face."], 96.0, delta=1.0
            )
            self.assertAlmostEqual(
                values["Use very dim lighting on the face."], 48.0, delta=1.0
            )
            if shot != "11":
                self.assertAlmostEqual(
                    values["Use bright, even lighting on the face."], 160.0, delta=1.0
                )

    def test_source_links_and_synthetic_caches_load_without_geometry_or_identity_changes(
        self,
    ):
        self.prepare()
        config = SimpleNamespace(image_size=32, align_paired_targets=True)
        dataset = FFHQDataset(
            self.output,
            split="train",
            precompute_3d=False,
            config=config,
            transform=create_transforms(32, "val"),
            rendered_transform=create_rendered_transform(8),
        )
        self.assertEqual(len(dataset), 10)
        row = next(
            row
            for row in self.rows("train")
            if row["source"] == "fei-001-11.png"
            and row["text"] == "Use dim lighting on the face."
        )
        original_image = self.root / "images/train" / row["source"]
        linked_image = self.output / "images/train" / row["source"]
        self.assertEqual(original_image.stat().st_ino, linked_image.stat().st_ino)
        original_cache = self.cache_path(self.root, "train", row["source"])
        linked_cache = self.cache_path(self.output, "train", row["source"])
        self.assertEqual(original_cache.stat().st_ino, linked_cache.stat().st_ino)
        with np.load(original_cache) as original, np.load(
            self.cache_path(self.output, "train", row["target"])
        ) as target:
            np.testing.assert_array_equal(
                target["params"][:-27], original["params"][:-27]
            )
            np.testing.assert_array_equal(target["alignment"], original["alignment"])
            np.testing.assert_array_equal(
                target["rendered"], np.full((8, 8, 3), 60, np.uint8)
            )
            np.testing.assert_allclose(
                target["params"][-27:][[0, 9, 18]], -0.2, atol=1e-6
            )
            np.testing.assert_allclose(
                np.delete(target["params"][-27:], [0, 9, 18]), 0.12, atol=1e-6
            )
        for item in dataset:
            self.assertEqual(tuple(item["image"].shape), (3, 32, 32))

    def test_missing_or_corrupt_cache_skips_only_affected_synthetic_sources(self):
        self.cache_path(self.root, "train", "fei-001-13.png").unlink()
        self.cache_path(self.root, "val", "fei-002-14.png").write_bytes(b"invalid npz")
        self.prepare()
        for split, forbidden in (
            ("train", "fei-001-13.png"),
            ("val", "fei-002-14.png"),
        ):
            rows = self.rows(split)
            self.assertEqual(rows[:2], self.original_rows[split][:2])
            self.assertEqual(len(rows), 7)
            self.assertTrue(all(row["source"] != forbidden for row in rows[2:]))
        manifest = json.loads((self.output / "manifest.json").read_text())
        skipped = {(row["split"], row["source"]) for row in manifest["skipped"]}
        self.assertIn(("train", "fei-001-13.png"), skipped)
        self.assertIn(("val", "fei-002-14.png"), skipped)

    def test_missing_reference_cache_skips_identity_lighting_without_reestimating(self):
        self.cache_path(self.root, "train", "fei-001-11.png").unlink()
        self.prepare()
        self.assertEqual(self.rows("train"), self.original_rows["train"][:2])

    def test_manifest_records_provenance_and_original_data_remain_unchanged(self):
        before = self.snapshot()
        report = self.prepare()
        manifest = json.loads((self.output / "manifest.json").read_text())
        self.assertEqual(report, manifest)
        self.assertEqual(manifest["source_root"], str(self.root.resolve()))
        self.assertEqual(manifest["splits"]["train"]["synthetic_pairs"], 8)
        self.assertEqual(manifest["splits"]["train"]["removed_lighting_pairs"], 3)
        self.assertEqual(len(manifest["pairs"]), 16)
        self.assertEqual(
            {row["state"] for row in manifest["pairs"]}, {"bright", "dim", "very_dim"}
        )
        row = next(
            row
            for row in manifest["pairs"]
            if row["source"] == "fei-001-11.png" and row["state"] == "dim"
        )
        self.assertAlmostEqual(row["gain"], 0.6, places=6)
        self.assertAlmostEqual(row["reference_face_luminance"], 160 / 255, places=6)
        self.assertEqual(self.snapshot(), before)

    def test_existing_output_and_output_inside_original_are_rejected_without_changes(
        self,
    ):
        before = self.snapshot()
        self.output.mkdir()
        marker = self.output / "preserve.txt"
        marker.write_text("existing output")
        with self.assertRaises(FileExistsError):
            self.prepare()
        self.assertEqual(marker.read_text(), "existing output")
        self.output = self.root / "accidental_derived"
        with self.assertRaises(ValueError):
            self.prepare()
        self.assertFalse(self.output.exists())
        self.assertEqual(self.snapshot(), before)


if __name__ == "__main__":
    unittest.main()
