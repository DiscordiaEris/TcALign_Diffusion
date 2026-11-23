import csv
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from data import datasets
from models.text_conditional_3d_editor import TextConditional3DEditor


class InstructionAugmentationTest(unittest.TestCase):
    def setUp(self):
        self.bank = getattr(datasets, "INSTRUCTION_PARAPHRASES", None)
        self.assertIsNotNone(self.bank, "Paired instructions need semantic-preserving variants")

    def test_paraphrases_preserve_attribute_direction_and_intensity(self):
        config = SimpleNamespace(shape_dim=80, expression_dim=64, pose_dim=3,
                                 texture_dim=80, total_3d_dim=254, clip_embed_dim=768)
        editor = TextConditional3DEditor(config)
        seen = set()
        for canonical, alternatives in self.bank.items():
            self.assertGreaterEqual(len(alternatives), 3)
            expected_attribute = editor.parse_attributes(canonical)
            canonical_words = set(re.findall(r"[a-z]+", canonical.lower()))
            for instruction in (canonical, *alternatives):
                self.assertNotIn(instruction, seen)
                seen.add(instruction)
                self.assertEqual(editor.parse_attributes(instruction), expected_attribute)
                words = set(re.findall(r"[a-z]+", instruction.lower()))
                if canonical_words.intersection({"left", "right"}):
                    direction = "left" if "left" in canonical_words else "right"
                    opposite = "right" if direction == "left" else "left"
                    self.assertIn(direction, words)
                    self.assertNotIn(opposite, words)
                    self.assertEqual("slightly" in words, "slightly" in canonical_words)
                elif "lighting" in canonical:
                    self.assertEqual("bright" in words, "bright" in canonical_words)
                    self.assertEqual("dim" in words, "dim" in canonical_words)
                    self.assertEqual("very" in words, "very" in canonical_words)
                elif "neutral" in canonical_words:
                    self.assertIn("neutral", words)
                    self.assertFalse(words.intersection({"smile", "smiling", "happy"}))
                elif "smile" in canonical_words:
                    self.assertTrue(words.intersection({"smile", "smiling", "happy"}))
                    self.assertNotIn("neutral", words)

    def test_unknown_instructions_are_preserved(self):
        instruction = "Keep the person's glasses while changing the background."
        self.assertEqual(datasets.augment_instruction(instruction), instruction)

    def test_training_augments_without_changing_validation_or_pair_metadata(self):
        canonical = "Turn the head toward the right side of the image."
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split, enabled in (("train", True), ("val", True), ("train", False)):
                config = SimpleNamespace(image_size=8, augment_instructions=enabled)
                images = root / "images" / split
                images.mkdir(parents=True, exist_ok=True)
                for name in ("source.png", "target.png"):
                    Image.new("RGB", (8, 8)).save(images / name)
                with (root / f"metadata_{split}.csv").open("w", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(("source", "target", "text"))
                    writer.writerow(("source.png", "target.png", canonical))
                cache_dir = root / "deep3d_params" / "aligned_sh_v2_8" / split
                cache_dir.mkdir(parents=True, exist_ok=True)
                provisional = datasets.FFHQDataset.__new__(datasets.FFHQDataset)
                provisional.image_dir = images
                provisional.cache_dir = cache_dir
                for name in ("source.png", "target.png"):
                    np.savez(provisional._cache_path(name), params=np.zeros(254, np.float32),
                             rendered=np.zeros((8, 8, 3), np.uint8),
                             alignment=np.eye(2, 3, dtype=np.float32))
                dataset = datasets.FFHQDataset(
                    root, split=split, config=config, precompute_3d=False,
                    rendered_transform=lambda image: torch.zeros(3, 8, 8))
                instructions = {dataset[0]["text"] for _ in range(128)}
                if split == "train" and enabled:
                    self.assertGreater(len(instructions), 1)
                    self.assertTrue(instructions.issubset({canonical, *self.bank[canonical]}))
                else:
                    self.assertEqual(instructions, {canonical})
                self.assertEqual(dataset.metadata, [("source.png", "target.png", canonical)])


if __name__ == "__main__":
    unittest.main()
