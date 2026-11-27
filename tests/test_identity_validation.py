import importlib
import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch


try:
    validation = importlib.import_module("utils.identity_validation")
except ModuleNotFoundError:
    validation = None


EDITS = (
    ("left", "Turn the head toward the left side of the image.", -0.4),
    ("right", "Turn the head slightly toward the right side of the image.", 0.4),
    ("smile", "Make the person smile.", 0.1),
    ("dim", "Use dim lighting on the face.", -0.45),
    ("very_dim", "Use very dim lighting on the face.", -0.7),
)


def portrait(identity, value=0.0):
    image = torch.full((3, 8, 8), value)
    image[2, 0, 0] = identity / 20
    return image


class PairedDataset:
    def __init__(self, identities=(8, 3, 7, 2, 5, 6)):
        self.metadata = []
        self.samples = []
        self.add_subject(99, frontal=False)
        self.add_subject(1, missing="very_dim")
        for identity in identities:
            self.add_subject(identity)

    def add_subject(self, identity, missing=None, frontal=True):
        source = f"fei-{identity:03d}-{'11' if frontal else '03'}.jpg"
        for edit, text, value in EDITS:
            if edit == missing:
                continue
            target = f"fei-{identity:03d}-{edit}.png"
            self.metadata.append((source, target, text))
            self.samples.append({
                "image": portrait(identity), "target_image": portrait(identity, value),
                "params": torch.tensor([identity, 0.0, 0.0]),
                "source_alignment": torch.eye(2, 3),
                "target_alignment": torch.eye(2, 3),
                "rendered_image": portrait(identity), "text": text,
            })

    def __getitem__(self, index):
        return self.samples[index]


class PoseProcessor:
    def reconstruct_aligned(self, image, alignment):
        values = image.new_zeros(len(image), 3)
        values[:, 1] = image[:, 0, 0, 0]
        return values


class SampledModel:
    def __init__(self, *, reverse_pose=False, copy_exposure=False):
        self.config = SimpleNamespace(guidance_scale=1.5, inference_strength=0.65)
        self.deep3d_processor = PoseProcessor()
        self.text_3d_editor = SimpleNamespace(attribute_slices={"pose": slice(0, 3)})
        self.identity_loss = None
        self.training = True
        self.calls = []
        self.reverse_pose = reverse_pose
        self.copy_exposure = copy_exposure

    def generate(self, images, texts, **kwargs):
        self.calls.append((images.clone(), texts[:], kwargs))
        identity = round(float(images[0, 2, 0, 0]) * 20)
        text = texts[0]
        if "left" in text:
            value = 0.4 if self.reverse_pose else -0.4
        elif "right" in text:
            value = -0.4 if self.reverse_pose else 0.4
        elif "smile" in text:
            value = 0.1
        else:
            value = 0.0 if self.copy_exposure else (-0.7 if "very dim" in text else -0.45)
        return portrait(identity, value)[None], None


class IndependentIdentity:
    def __init__(self, fail=None):
        self.fail = fail

    def encode(self, image):
        value = round(float(image[0, 0, 0, 0]), 2)
        identity = round(float(image[0, 2, 0, 0]) * 20)
        return None if (identity, value) == self.fail else (identity, value)

    def similarity(self, source, edited):
        if source is None or edited is None:
            return None
        return 0.8 if edited[1] > -0.45 else 0.05


class OptionalFaceNet:
    def __init__(self):
        self.normalizations = []

    def embed(self, images, alignment, *, normalize_exposure=False):
        self.normalizations.append(normalize_exposure)
        return torch.stack((torch.ones(len(images)), images[:, 0, 0, 0]), dim=1)


class IdentityValidationTest(unittest.TestCase):
    def evaluator(self):
        self.assertIsNotNone(validation, "Sampled identity validation module is required")
        return validation.evaluate_sampled_identity

    def gate(self):
        self.assertIsNotNone(validation, "Sampled identity checkpoint gate is required")
        return validation.should_replace_identity_checkpoint

    def run_panel(self, **kwargs):
        return self.evaluator()(SampledModel(), PairedDataset(), "cpu",
                                evaluator=IndependentIdentity(), **kwargs)

    def test_panel_uses_complete_frontal_subjects_in_metadata_order(self):
        model = SampledModel()
        state = torch.get_rng_state().clone()
        result = self.evaluator()(model, PairedDataset(), "cpu", evaluator=IndependentIdentity())
        self.assertEqual(len(result["rows"]), 30)
        self.assertEqual([row["source"] for row in result["rows"]][::5],
                         [f"fei-{identity:03d}-11.jpg" for identity in (8, 3, 7, 2, 5, 6)])
        self.assertEqual([row["edit"] for row in result["rows"][:5]],
                         ["left", "right", "smile", "dim", "very_dim"])
        self.assertTrue(model.training)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertEqual(len(model.calls), 30)
        for images, texts, options in model.calls:
            self.assertEqual(len(images), 1)
            self.assertEqual(options["num_inference_steps"], 50)
            self.assertEqual(options["guidance_scale"], 1.5)
            self.assertEqual(options["strength"], 0.65)
            self.assertFalse(options["estimate_final_theta"])
            self.assertEqual(options["generator"].initial_seed(), 42)
            self.assertEqual(tuple(options["theta_0"].shape), (1, 3))
            self.assertEqual(tuple(options["source_alignment"].shape), (1, 2, 3))
            self.assertEqual(tuple(options["source_rendered"].shape), (1, 3, 8, 8))
        self.assertEqual(model.calls[3][1], ["Dim the lighting on the face."])
        self.assertEqual(model.calls[4][1], ["Make the lighting on the face very dim."])
        self.assertEqual(model.calls[1][1], ["Turn the head slightly toward the right side of the image."])

    def test_identity_selection_ignores_lighting_and_reports_all_coverage(self):
        metrics = self.run_panel()["aggregate"]
        self.assertAlmostEqual(metrics["identity_score"], 0.8)
        for key, count in (("identity", 18), ("pose", 12), ("exposure", 12)):
            self.assertEqual(metrics[f"{key}_expected"], count)
            self.assertEqual(metrics[f"{key}_valid"], count)
        self.assertEqual(metrics["pose_direction_hits"], 12)
        self.assertAlmostEqual(metrics["pose_yaw_abs_error_deg"], 0.0)
        self.assertAlmostEqual(metrics["exposure_error"], 0.0)
        self.assertTrue(self.gate()(metrics, best_score=0.7))

    def test_detection_failure_remains_visible_and_blocks_checkpoint(self):
        result = self.evaluator()(SampledModel(), PairedDataset(), "cpu",
                                  evaluator=IndependentIdentity(fail=(8, -0.4)))
        self.assertEqual(result["aggregate"]["identity_valid"], 17)
        self.assertIsNone(result["rows"][0]["sface_cosine"])
        self.assertFalse(self.gate()(result["aggregate"], 0.7))

    def test_missing_complete_subjects_blocks_checkpoint_with_expected_counts(self):
        result = self.evaluator()(SampledModel(), PairedDataset((8, 3, 7, 2, 5)), "cpu",
                                  evaluator=IndependentIdentity())
        self.assertEqual(result["aggregate"]["subjects_selected"], 5)
        self.assertEqual(result["aggregate"]["identity_expected"], 18)
        self.assertEqual(result["aggregate"]["identity_valid"], 15)
        self.assertFalse(self.gate()(result, 0.7))

    def test_wrong_direction_and_source_copy_exposure_block_checkpoint(self):
        for options, key in (({"reverse_pose": True}, "pose_direction_hits"),
                             ({"copy_exposure": True}, "exposure_error")):
            with self.subTest(options=options):
                result = self.evaluator()(SampledModel(**options), PairedDataset(), "cpu",
                                          evaluator=IndependentIdentity())
                self.assertFalse(self.gate()(result, 0.7))
                if key == "pose_direction_hits":
                    self.assertEqual(result["aggregate"][key], 0)
                else:
                    self.assertGreater(result["aggregate"][key], 0.1)

    def test_optional_facenet_records_raw_and_normalized_scores(self):
        model = SampledModel()
        model.identity_loss = OptionalFaceNet()
        result = self.evaluator()(model, PairedDataset(), "cpu", evaluator=IndependentIdentity())
        self.assertIn("facenet_raw_cosine", result["rows"][0])
        self.assertIn("facenet_normalized_cosine", result["rows"][0])
        self.assertEqual(set(model.identity_loss.normalizations), {True, False})

    def test_default_independent_evaluator_uses_local_weights(self):
        self.evaluator()
        model = SampledModel()
        with patch.object(validation, "SFaceEvaluator", return_value=IndependentIdentity()) as factory:
            self.evaluator()(model, PairedDataset(), "cpu", subjects=1)
        self.assertIs(factory.call_args.args[0], model.deep3d_processor)
        self.assertTrue(str(factory.call_args.args[1]).endswith("models/face_recognition_sface_2021dec.onnx"))

    def test_checkpoint_gate_rejects_bad_or_missing_metrics(self):
        metrics = self.run_panel()["aggregate"]
        for changes in ({"identity_score": math.nan}, {"identity_score": math.inf},
                        {"pose_yaw_abs_error_deg": 8.01}, {"exposure_error": 0.101},
                        {"identity_valid": 17}, {"pose_valid": 11},
                        {"pose_direction_hits": 11}, {"exposure_valid": 11},
                        {"identity_expected": 15}):
            with self.subTest(changes=changes):
                self.assertFalse(self.gate()({**metrics, **changes}, 0.7))
        self.assertFalse(self.gate()({}, 0.7))
        self.assertFalse(self.gate()(metrics, 0.8))
        self.assertFalse(self.gate()(metrics, 0.7999))
        self.assertTrue(self.gate()(metrics, 0.7998))

    def test_checkpoint_gate_accepts_first_eligible_model_and_inclusive_error_limits(self):
        metrics = self.run_panel()["aggregate"]
        self.assertTrue(self.gate()(metrics, None))
        self.assertTrue(self.gate()(metrics, -math.inf))
        self.assertTrue(self.gate()({**metrics, "pose_yaw_abs_error_deg": 8.0,
                                    "exposure_error": 0.1}, 0.7))
        self.assertFalse(self.gate()(metrics, math.nan))
        self.assertFalse(self.gate()(metrics, math.inf))


if __name__ == "__main__":
    unittest.main()
