import math
import unittest
from types import SimpleNamespace

import torch

from utils.identity_validation import (
    evaluate_sampled_identity,
    should_replace_identity_checkpoint,
)


_EDITS = (
    ("left", "Turn the head toward the left side of the image.", -0.4),
    ("right", "Turn the head slightly toward the right side of the image.", 0.4),
    ("smile", "Make the person smile.", 0.1),
    ("dim", "Use dim lighting on the face.", -0.45),
    ("very_dim", "Use very dim lighting on the face.", -0.7),
)


def portrait(value=0.0, expression=0.0):
    image = torch.full((3, 8, 8), value)
    image[1, 0, 0] = expression
    return image


class ExpressionDataset:
    def __init__(self, displacement=0.5, subjects=6):
        self.metadata = []
        self.samples = []
        for subject in range(subjects):
            for edit, prompt, value in _EDITS:
                source = f"fei-{subject:03d}-11.jpg"
                target = f"fei-{subject:03d}-{edit}.png"
                self.metadata.append((source, target, prompt))
                self.samples.append({
                    "image": portrait(),
                    "target_image": portrait(
                        value, displacement if edit == "smile" else 0.0
                    ),
                    "source_alignment": torch.eye(2, 3),
                    "target_alignment": torch.eye(2, 3),
                    "params": torch.zeros(5),
                    "rendered_image": portrait(),
                })

    def __getitem__(self, index):
        return self.samples[index]


class ExpressionProcessor:
    def reconstruct_aligned(self, image, alignment):
        coefficients = image.new_zeros(len(image), 5)
        coefficients[:, 1] = image[:, 0, 0, 0]
        orthogonal = image[:, 1, 0, 1] - image[:, 0, 0, 0]
        coefficients[:, 3] = image[:, 1, 0, 0] + orthogonal
        coefficients[:, 4] = -image[:, 1, 0, 0] + orthogonal
        return coefficients


class ExpressionModel:
    def __init__(self, prediction=0.5, orthogonal=0.0):
        self.config = SimpleNamespace(guidance_scale=1.5, inference_strength=0.65)
        self.deep3d_processor = ExpressionProcessor()
        self.text_3d_editor = SimpleNamespace(attribute_slices={
            "pose": slice(0, 3), "expression": slice(3, 5)
        })
        self.identity_loss = None
        self.prediction = prediction
        self.orthogonal = orthogonal

    def generate(self, source, texts, **kwargs):
        prompt = texts[0]
        if "left" in prompt:
            value = -0.4
        elif "right" in prompt:
            value = 0.4
        elif "smile" in prompt:
            image = portrait(0.1, self.prediction)
            image[1, 0, 1] = 0.1 + self.orthogonal
            return image[None], None
        else:
            value = -0.7 if "very dim" in prompt else -0.45
        return portrait(value)[None], None


class IndependentIdentity:
    def encode(self, image):
        return 1

    def similarity(self, source, generated):
        return 0.9


class IdentityExpressionGateTest(unittest.TestCase):
    def panel(self, prediction=0.5, displacement=0.5, subjects=6, orthogonal=0.0):
        return evaluate_sampled_identity(
            ExpressionModel(prediction, orthogonal),
            ExpressionDataset(displacement, subjects),
            "cpu",
            evaluator=IndependentIdentity(),
        )

    def test_target_expression_has_zero_error_with_six_valid_smiles(self):
        result = self.panel()
        aggregate = result["aggregate"]
        self.assertEqual(aggregate["expression_expected"], 6)
        self.assertEqual(aggregate["expression_valid"], 6)
        self.assertEqual(aggregate["expression_error_ratio"], 0.0)
        for row in result["rows"]:
            if row["edit"] == "smile":
                self.assertTrue(row["expression_valid"])
                self.assertEqual(row["expression_error_ratio"], 0.0)
            else:
                self.assertNotIn("expression_error_ratio", row)
        self.assertTrue(should_replace_identity_checkpoint(
            result, 0.8, require_expression=True
        ))

    def test_copying_source_expression_has_unit_error_and_blocks_promotion(self):
        result = self.panel(prediction=0.0)
        self.assertEqual(result["aggregate"]["expression_error_ratio"], 1.0)
        self.assertFalse(should_replace_identity_checkpoint(
            result, 0.8, require_expression=True
        ))

    def test_partial_expression_reports_normalized_squared_error(self):
        aggregate = self.panel(prediction=0.25)["aggregate"]
        self.assertEqual(aggregate["expression_error_ratio"], 0.25)
        self.assertTrue(should_replace_identity_checkpoint(
            aggregate, 0.8, require_expression=True
        ))

    def test_tiny_expression_displacements_use_the_control_loss_floor(self):
        aggregate = self.panel(prediction=0.0, displacement=0.01)["aggregate"]
        self.assertAlmostEqual(aggregate["expression_error_ratio"], 0.04, places=6)

    def test_nonfinite_expression_stays_missing_and_cannot_pass(self):
        result = self.panel(prediction=math.nan)
        aggregate = result["aggregate"]
        self.assertEqual(aggregate["expression_valid"], 0)
        self.assertIsNone(aggregate["expression_error_ratio"])
        smile = next(row for row in result["rows"] if row["edit"] == "smile")
        self.assertFalse(smile["expression_valid"])
        self.assertIsNone(smile["expression_error_ratio"])
        self.assertFalse(should_replace_identity_checkpoint(
            result, 0.8, require_expression=True
        ))

    def test_expression_gate_rejects_bad_errors_and_incomplete_coverage(self):
        metrics = self.panel()["aggregate"]
        for change in (
            {"expression_expected": 5}, {"expression_valid": 5},
            {"expression_valid": 0}, {"expression_error_ratio": None},
            {"expression_error_ratio": math.nan},
            {"expression_error_ratio": math.inf},
            {"expression_error_ratio": -0.1},
            {"expression_error_ratio": 1.0},
            {"expression_error_ratio": 1.1},
        ):
            with self.subTest(change=change):
                self.assertFalse(should_replace_identity_checkpoint(
                    {**metrics, **change}, 0.8, require_expression=True
                ))
        without_expression = {
            name: value for name, value in metrics.items()
            if not name.startswith("expression_")
        }
        self.assertFalse(should_replace_identity_checkpoint(
            without_expression, 0.8, require_expression=True
        ))
        self.assertFalse(should_replace_identity_checkpoint(
            self.panel(subjects=5), 0.8, require_expression=True
        ))

    def test_stricter_expression_limit_remains_below_copy_error(self):
        metrics = self.panel(prediction=0.25)["aggregate"]
        self.assertTrue(should_replace_identity_checkpoint(
            metrics, 0.8, require_expression=True, max_expression_error=0.3
        ))
        self.assertFalse(should_replace_identity_checkpoint(
            metrics, 0.8, require_expression=True, max_expression_error=0.25
        ))
        for limit in (math.nan, math.inf, -0.1, 0.0):
            with self.subTest(limit=limit):
                self.assertFalse(should_replace_identity_checkpoint(
                    metrics, 0.8, require_expression=True, max_expression_error=limit
                ))
        self.assertFalse(should_replace_identity_checkpoint(
            self.panel(prediction=0.0), 0.8,
            require_expression=True, max_expression_error=2.0
        ))

    def test_optional_expression_gate_preserves_legacy_selection(self):
        metrics = self.panel(prediction=0.0)["aggregate"]
        self.assertTrue(should_replace_identity_checkpoint(metrics, 0.8))
        legacy = {
            name: value for name, value in metrics.items()
            if not name.startswith("expression_")
        }
        self.assertTrue(should_replace_identity_checkpoint(legacy, 0.8))
        self.assertTrue(should_replace_identity_checkpoint(
            {**metrics, "expression_error_ratio": math.nan}, 0.8,
            require_expression=False
        ))
        self.assertFalse(should_replace_identity_checkpoint(
            {**metrics, "pose_direction_hits": 11}, 0.8, require_expression=True
        ))

    def test_orthogonal_expression_error_above_one_can_preserve_baseline_progress(self):
        result = self.panel(orthogonal=0.75)
        aggregate = result["aggregate"]
        self.assertAlmostEqual(aggregate["expression_error_ratio"], 2.25)
        self.assertEqual(aggregate["expression_progress_valid"], 6)
        self.assertEqual(aggregate["expression_direction_hits"], 6)
        self.assertEqual(aggregate["expression_progress"], 1.0)
        self.assertTrue(should_replace_identity_checkpoint(
            result, 0.8, require_expression=True,
            max_expression_error=2.5, min_expression_progress=0.9,
        ))
        self.assertFalse(should_replace_identity_checkpoint(
            result, 0.8, require_expression=True, max_expression_error=2.5,
        ))
        self.assertFalse(should_replace_identity_checkpoint(
            result, 0.8, require_expression=True,
            max_expression_error=2.2, min_expression_progress=0.9,
        ))

    def test_progress_is_clipped_per_subject_and_one_source_copy_is_rejected(self):
        class OneCopyModel(ExpressionModel):
            smiles = 0

            def generate(self, source, texts, **kwargs):
                if "smile" in texts[0]:
                    self.prediction = 0.0 if self.smiles == 0 else 0.75
                    self.smiles += 1
                return super().generate(source, texts, **kwargs)

        result = evaluate_sampled_identity(
            OneCopyModel(), ExpressionDataset(), "cpu", evaluator=IndependentIdentity(),
        )
        aggregate = result["aggregate"]
        self.assertEqual(aggregate["expression_progress_valid"], 6)
        self.assertEqual(aggregate["expression_direction_hits"], 5)
        self.assertAlmostEqual(aggregate["expression_progress"], 5 / 6)
        self.assertFalse(should_replace_identity_checkpoint(
            result, 0.8, require_expression=True,
            max_expression_error=2.0, min_expression_progress=0.8,
        ))

    def test_near_zero_target_displacement_has_no_valid_progress(self):
        result = self.panel(prediction=0.0001, displacement=0.0001)
        self.assertEqual(result["aggregate"]["expression_progress_valid"], 0)
        self.assertIsNone(result["aggregate"]["expression_progress"])
        smile = next(row for row in result["rows"] if row["edit"] == "smile")
        self.assertIsNone(smile["expression_progress"])
        self.assertFalse(should_replace_identity_checkpoint(
            result, 0.8, require_expression=True,
            max_expression_error=1.0, min_expression_progress=0.1,
        ))

    def test_baseline_progress_gate_rejects_regression_and_bad_coverage(self):
        aggregate = self.panel(prediction=0.4)["aggregate"]
        self.assertAlmostEqual(aggregate["expression_progress"], 0.8)
        self.assertTrue(should_replace_identity_checkpoint(
            aggregate, 0.8, require_expression=True,
            max_expression_error=0.1, min_expression_progress=0.8,
        ))
        for changes in (
            {"expression_progress": 0.71}, {"expression_progress": None},
            {"expression_progress": math.nan}, {"expression_progress": math.inf},
            {"expression_progress_valid": 5}, {"expression_direction_hits": 5},
        ):
            with self.subTest(changes=changes):
                self.assertFalse(should_replace_identity_checkpoint(
                    {**aggregate, **changes}, 0.8, require_expression=True,
                    max_expression_error=0.1, min_expression_progress=0.72,
                ))


if __name__ == "__main__":
    unittest.main()
