"""CPU contracts for the frozen ArcFace loss; no pretrained downloads needed."""

import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import onnx
import torch
from onnx import TensorProto, helper, numpy_helper

from models.arcface_identity_loss import (
    ArcFaceIdentityLoss,
    FrozenArcFaceEncoder,
    FaceDetectionError,
    align_arcface_crops,
)


class FixedLandmarkProcessor:
    def __init__(self, *, failures=()):
        self.failures = set(failures)
        self.calls = 0

    def detect_yunet_face(self, rgb):
        if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8:
            raise AssertionError("Detection must receive detached uint8 RGB")
        self.calls += 1
        if self.calls in self.failures:
            return None
        return np.array([
            0, 0, 112, 112,
            38.2946, 51.6963, 73.5318, 51.5014, 56.0252, 71.7366,
            41.5493, 92.3655, 70.7299, 92.2041, .99,
        ], dtype=np.float32)


class PixelLandmarkProcessor(FixedLandmarkProcessor):
    """Translate landmarks by a deterministic marker in detached RGB input."""

    def __init__(self):
        super().__init__()
        self.seen = []

    def detect_yunet_face(self, rgb):
        self.seen.append(rgb.copy())
        face = super().detect_yunet_face(rgb)
        face[4:14].reshape(5, 2)[:, 0] += int(rgb[0, 0, 0]) % 17
        return face


def write_six_op_fixture(path):
    rng = np.random.default_rng(21)
    arrays = {
        "conv_w": rng.normal(0, .25, (2, 3, 1, 1)).astype(np.float32),
        "conv_b": np.array([-.2, .1], dtype=np.float32),
        "slope": np.array([.25, .4], dtype=np.float32).reshape(2, 1, 1),
        "scale": np.array([.9, 1.1], dtype=np.float32),
        "bias": np.array([.05, -.03], dtype=np.float32),
        "mean": np.array([.1, -.15], dtype=np.float32),
        "var": np.array([.75, 1.3], dtype=np.float32),
        "fc_w": rng.normal(0, .2, (512, 32)).astype(np.float32),
        "fc_b": rng.normal(0, .1, (512,)).astype(np.float32),
    }
    nodes = [
        helper.make_node("Conv", ["input.1", "conv_w", "conv_b"], ["conv"],
                         kernel_shape=[1, 1], strides=[28, 28]),
        helper.make_node("PRelu", ["conv", "slope"], ["prelu"]),
        helper.make_node("BatchNormalization", ["prelu", "scale", "bias", "mean", "var"],
                         ["bn"], epsilon=1e-5),
        helper.make_node("Add", ["conv", "bn"], ["residual"]),
        helper.make_node("Flatten", ["residual"], ["flat"], axis=1),
        helper.make_node("Gemm", ["flat", "fc_w", "fc_b"], ["683"], transB=1),
    ]
    graph = helper.make_graph(
        nodes, "arcface_cpu_contract",
        [helper.make_tensor_value_info("input.1", TensorProto.FLOAT, [None, 3, 112, 112])],
        [helper.make_tensor_value_info("683", TensorProto.FLOAT, [None, 512])],
        [numpy_helper.from_array(value, name) for name, value in arrays.items()],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)], ir_version=6)
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ArcFaceIdentityLossTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.path = Path(cls.directory.name) / "tiny_arcface.onnx"
        cls.digest = write_six_op_fixture(cls.path)
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def make_loss(self, processor=None, **kwargs):
        with patch("models.arcface_identity_loss.W600K_SHA256", self.digest):
            return ArcFaceIdentityLoss(self.path, processor or FixedLandmarkProcessor(), **kwargs)

    def test_six_ops_match_opencv_and_keep_input_gradient(self):
        encoder = FrozenArcFaceEncoder(self.path, expected_sha256=self.digest)
        image = torch.randn(2, 3, 112, 112, generator=torch.Generator().manual_seed(8),
                            requires_grad=True)
        reference = cv2.dnn.readNetFromONNX(str(self.path))
        reference.setInput(image.detach().numpy())
        expected = torch.from_numpy(reference.forward().copy())
        actual = encoder(image)
        torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-6)
        actual.square().mean().backward()
        self.assertTrue(torch.isfinite(image.grad).all())
        self.assertGreater(float(image.grad.norm()), 0)
        encoder.train()
        self.assertFalse(encoder.training)
        self.assertFalse(any(parameter.requires_grad for parameter in encoder.parameters()))
        self.assertEqual(dict(encoder.state_dict()), {})

    def test_rejects_unexpected_weights_instead_of_silently_changing_identity_model(self):
        with self.assertRaisesRegex(ValueError, "SHA256"):
            FrozenArcFaceEncoder(self.path)

    def test_rejects_unsupported_graph_even_with_expected_hash(self):
        graph = onnx.load(self.path)
        graph.graph.node[0].op_type = "Relu"
        altered = Path(self.directory.name) / "unsupported.onnx"
        onnx.save(graph, altered)
        digest = hashlib.sha256(altered.read_bytes()).hexdigest()
        with self.assertRaisesRegex(ValueError, "Unsupported.*Relu"):
            FrozenArcFaceEncoder(altered, expected_sha256=digest)

    def test_source_only_loss_ignores_target_and_reaches_generated_pixels(self):
        loss = self.make_loss()
        rng = torch.Generator().manual_seed(12)
        source = (torch.rand(1, 3, 112, 112, generator=rng) * 1.2 - .6).requires_grad_()
        generated = (source.detach() + torch.randn(1, 3, 112, 112, generator=rng) * .2).requires_grad_()
        target = torch.zeros_like(source, requires_grad=True)
        value = loss(generated, source, target, None, None)
        other_target = loss(generated, source, -target.detach() + .9, None, None)
        other_source = loss(generated, -source.detach(), target, None, None)
        torch.testing.assert_close(value, other_target)
        self.assertGreater(abs(float(value - other_source)), .01)
        value.backward()
        self.assertGreater(float(generated.grad.norm()), 0)
        self.assertTrue(torch.isfinite(generated.grad).all())
        self.assertIsNone(source.grad)
        self.assertIsNone(target.grad)
        loss.train()
        self.assertFalse(loss.training)
        self.assertFalse(loss.encoder.training)
        self.assertEqual(dict(loss.state_dict()), {})

    def test_embed_is_float32_under_outer_autocast(self):
        loss = self.make_loss()
        image = torch.rand(1, 3, 112, 112, requires_grad=True)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            embedding = loss.embed(image)
        self.assertEqual(embedding.dtype, torch.float32)
        torch.testing.assert_close(embedding.norm(dim=-1), torch.ones(1))
        self.assertEqual(loss.metric_name, "arcface")
        self.assertFalse(loss.supports_exposure_normalization)
        with self.assertRaisesRegex(ValueError, "exposure"):
            loss.embed(image, normalize_exposure=True)

    def test_generated_missing_detection_uses_source_affine_and_reports_count(self):
        processor = FixedLandmarkProcessor(failures={2})
        loss = self.make_loss(processor)
        source = torch.rand(1, 3, 112, 112)
        generated = torch.rand(1, 3, 112, 112, requires_grad=True)
        value = loss(generated, source, None, None, None)
        self.assertEqual(loss.last_detection_fallbacks, 1)
        value.backward()
        self.assertTrue(torch.isfinite(generated.grad).all())
        self.assertGreater(float(generated.grad.norm()), 0)
        loss(generated, source, None, None, None)
        self.assertEqual(loss.last_detection_fallbacks, 0)

    def test_missing_reference_or_eval_detection_raises(self):
        image = torch.rand(1, 3, 112, 112)
        reference_missing = self.make_loss(FixedLandmarkProcessor(failures={1}))
        with self.assertRaisesRegex(FaceDetectionError, "detect"):
            reference_missing(image, image, None, None, None)
        eval_missing = self.make_loss(FixedLandmarkProcessor(failures={1}))
        with self.assertRaisesRegex(FaceDetectionError, "detect"):
            eval_missing.embed(image)

    def test_paired_target_affine_is_fixed_across_generated_pixel_changes(self):
        processor = PixelLandmarkProcessor()
        loss = self.make_loss(processor, training_alignment="paired_target")
        rng = torch.Generator().manual_seed(28)
        source = torch.rand(1, 3, 112, 112, generator=rng) * 1.2 - .6
        source[0, 0, 0, 0] = -1  # Marker 0 gives an identity source affine.
        target = torch.zeros_like(source)
        target[0, 0, 0, 0] = 5 / 127.5 - 1  # Marker 5 shifts target landmarks right.
        first = (torch.rand(1, 3, 112, 112, generator=rng) * 1.4 - .7).requires_grad_()
        second = -first.detach()
        crops = []
        hook = loss.encoder.register_forward_pre_hook(
            lambda module, args: crops.append(args[0].detach().clone())
        )
        value = loss(first, source, target, None, None)
        loss(second, source, target, None, None)
        hook.remove()
        # The marker defines input->canonical translation x=-5 independently
        # of either generated image's content or detected face geometry.
        target_affine = torch.tensor([[[1., 0, -5], [0, 1, 0]]])
        torch.testing.assert_close(crops[1], align_arcface_crops(first, target_affine),
                                   atol=3e-5, rtol=0)
        torch.testing.assert_close(crops[3], align_arcface_crops(second, target_affine),
                                   atol=3e-5, rtol=0)
        torch.testing.assert_close(crops[0], crops[2], atol=0, rtol=0)
        expected_detected = [source, target, source, target]
        self.assertEqual(len(processor.seen), len(expected_detected))
        for actual, image in zip(processor.seen, expected_detected):
            expected = ((image[0] + 1) * 127.5).round().byte().permute(1, 2, 0).numpy()
            np.testing.assert_array_equal(actual, expected)
        value.backward()
        self.assertTrue(torch.isfinite(first.grad).all())
        self.assertGreater(float(first.grad.norm()), 0)

    def test_paired_target_pixels_do_not_become_an_identity_reference(self):
        loss = self.make_loss(PixelLandmarkProcessor(), training_alignment="paired_target")
        rng = torch.Generator().manual_seed(29)
        source = (torch.rand(1, 3, 112, 112, generator=rng) * 1.2 - .6).requires_grad_()
        generated = (torch.rand(1, 3, 112, 112, generator=rng) * 1.2 - .6).requires_grad_()
        target = torch.zeros_like(source, requires_grad=True)
        other_target = torch.full_like(source, .9)
        other_target[0, 0, 0, 0] = 0  # Equal detector marker => identical landmarks.
        first = loss(generated, source, target, None, None)
        second = loss(generated, source, other_target, None, None)
        torch.testing.assert_close(first, second, atol=0, rtol=0)
        first.backward()
        self.assertIsNone(source.grad)
        self.assertIsNone(target.grad)
        self.assertGreater(float(generated.grad.norm()), 0)

    def test_paired_target_detection_failure_falls_back_to_source_and_resets_count(self):
        processor = FixedLandmarkProcessor(failures={2})
        loss = self.make_loss(processor, training_alignment="paired_target")
        source = torch.rand(1, 3, 112, 112)
        generated = torch.rand(1, 3, 112, 112, requires_grad=True)
        value = loss(generated, source, torch.zeros_like(source), None, None)
        self.assertEqual(loss.last_detection_fallbacks, 1)
        reference = self.make_loss()(generated, source, None, None, None)
        torch.testing.assert_close(value, reference, atol=0, rtol=0)
        value.backward()
        self.assertGreater(float(generated.grad.norm()), 0)
        loss(generated, source, torch.zeros_like(source), None, None)
        self.assertEqual(loss.last_detection_fallbacks, 0)

    def test_paired_target_missing_input_falls_back_but_source_or_eval_failure_raises(self):
        source = torch.rand(1, 3, 112, 112)
        generated = torch.rand(1, 3, 112, 112)
        loss = self.make_loss(training_alignment="paired_target")
        actual = loss(generated, source, None, None, None)
        self.assertEqual(loss.last_detection_fallbacks, 1)
        reference = self.make_loss()(generated, source, None, None, None)
        torch.testing.assert_close(actual, reference, atol=0, rtol=0)
        missing_source = self.make_loss(FixedLandmarkProcessor(failures={1}),
                                        training_alignment="paired_target")
        with self.assertRaises(FaceDetectionError):
            missing_source(generated, source, source, None, None)
        missing_eval = self.make_loss(FixedLandmarkProcessor(failures={1}),
                                      training_alignment="paired_target")
        with self.assertRaises(FaceDetectionError):
            missing_eval.embed(generated)

    def test_unknown_training_alignment_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "training_alignment"):
            self.make_loss(training_alignment="unknown")

    def test_real_yunet_detection_retains_pixel_gradients(self):
        root = Path(__file__).resolve().parents[1]
        detector = root / "models/face_detection_yunet_2023mar.onnx"
        portrait = root / "datasets/face_editing_exposure_20261123/images/val/fei-004-11.jpg"
        if not detector.is_file() or not portrait.is_file():
            self.skipTest("Local YuNet and FEI portrait are optional")
        from utils.deep3d_utils import Deep3DProcessor

        processor = Deep3DProcessor.__new__(Deep3DProcessor)
        processor.config = SimpleNamespace(face_detector_path=str(detector))
        rgb = cv2.cvtColor(cv2.imread(str(portrait)), cv2.COLOR_BGR2RGB)
        image = (torch.from_numpy(rgb).permute(2, 0, 1)[None].float() / 127.5 - 1).requires_grad_()
        loss = self.make_loss(processor)
        embedding = loss.embed(image)
        embedding[:, :10].sum().backward()
        self.assertTrue(torch.isfinite(image.grad).all())
        self.assertGreater(float(image.grad.norm()), 0)


class ArcFaceAlignmentTest(unittest.TestCase):
    def test_identity_translation_and_rotation_follow_opencv_pixel_coordinates(self):
        rng = np.random.default_rng(22)
        rgb = rng.uniform(0, 1, (112, 112, 3)).astype(np.float32)
        image = torch.from_numpy(rgb).permute(2, 0, 1)[None] * 2 - 1
        matrices = [
            np.array([[1, 0, 0], [0, 1, 0]], np.float32),
            np.array([[1, 0, 9], [0, 1, -5]], np.float32),
            np.array([[0, -1, 111], [1, 0, 0]], np.float32),
        ]
        for matrix in matrices:
            with self.subTest(matrix=matrix.tolist()):
                expected = cv2.warpAffine(rgb, matrix, (112, 112), borderMode=cv2.BORDER_CONSTANT)
                actual = align_arcface_crops(image, torch.from_numpy(matrix)[None])
                expected = torch.from_numpy(expected).permute(2, 0, 1)[None] * 2 - 1
                torch.testing.assert_close(actual, expected, atol=4e-5, rtol=0)

    def test_padding_is_rgb_black_and_affine_is_detached(self):
        image = torch.full((1, 3, 112, 112), .5, requires_grad=True)
        matrix = torch.tensor([[[1., 0, 3], [0, 1, 0]]], requires_grad=True)
        crop = align_arcface_crops(image, matrix)
        torch.testing.assert_close(crop[:, :, :, :3], torch.full_like(crop[:, :, :, :3], -1))
        crop.sum().backward()
        self.assertGreater(float(image.grad.norm()), 0)
        self.assertIsNone(matrix.grad)

    def test_singular_alignment_fails_with_clear_error(self):
        with self.assertRaisesRegex(ValueError, "invertible"):
            align_arcface_crops(torch.zeros(1, 3, 112, 112), torch.zeros(1, 2, 3))


if __name__ == "__main__":
    unittest.main()
