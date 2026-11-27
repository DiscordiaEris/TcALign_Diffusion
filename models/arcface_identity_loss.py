"""Frozen buffalo_l ArcFace supervision with differentiable RGB face crops.

The local w600k_r50 ONNX weights are external assets. Only its six inference
operators are supported, and the fingerprint is checked before loading them.
SFace remains a separate evaluator; this module never changes its weights or
preprocessing.
"""

import hashlib
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


W600K_SHA256 = "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43"
ARCFACE_LANDMARKS = np.array([
    [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
    [41.5493, 92.3655], [70.7299, 92.2041],
], dtype=np.float32)


class FaceDetectionError(ValueError):
    """A face has no valid detected five-point alignment for ArcFace scoring."""


class FrozenArcFaceEncoder(nn.Module):
    """Execute the verified six-op ONNX inference graph using frozen tensors.

    ``expected_sha256`` permits verifying small local contract fixtures. The
    training loss always uses the fingerprint of the shipped w600k_r50 model.
    Buffers follow the module's device but are excluded from training checkpoints.
    """

    _supported_ops = {"Conv", "PRelu", "BatchNormalization", "Add", "Flatten", "Gemm"}

    def __init__(self, weights_path, *, expected_sha256=None):
        super().__init__()
        path = Path(weights_path)
        if not path.is_file():
            raise FileNotFoundError(f"ArcFace ONNX weights not found: {path}")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        expected = expected_sha256 or W600K_SHA256
        if digest != expected:
            raise ValueError(f"ArcFace SHA256 mismatch: expected {expected}, received {digest}")

        import onnx
        from onnx import numpy_helper

        model = onnx.load_model(str(path), load_external_data=False)
        for node in model.graph.node:
            if node.op_type not in self._supported_ops or node.domain not in ("", "ai.onnx"):
                raise ValueError(f"Unsupported ArcFace ONNX operator: {node.domain}:{node.op_type}")
            if len(node.output) != 1:
                raise ValueError("ArcFace inference nodes must have exactly one output")
        onnx.checker.check_model(model)
        if [(op.domain, op.version) for op in model.opset_import] != [("", 11)]:
            raise ValueError("ArcFace requires the verified ONNX opset 11 graph")

        self._weight_names = {}
        for index, tensor in enumerate(model.graph.initializer):
            array = numpy_helper.to_array(tensor)
            if array.dtype != np.float32:
                raise ValueError(f"ArcFace initializer must be float32: {tensor.name}")
            name = f"weight_{index}"
            self._weight_names[tensor.name] = name
            self.register_buffer(name, torch.from_numpy(array.copy()), persistent=False)

        inputs = [value for value in model.graph.input if value.name not in self._weight_names]
        if len(inputs) != 1 or len(model.graph.output) != 1:
            raise ValueError("ArcFace must have one image input and one embedding output")
        input_type = inputs[0].type.tensor_type
        shape = [dim.dim_value for dim in input_type.shape.dim]
        if input_type.elem_type != onnx.TensorProto.FLOAT or shape[1:] != [3, 112, 112]:
            raise ValueError("ArcFace expects float32 RGB input [batch, 3, 112, 112]")
        self._input_name = inputs[0].name
        self._output_name = model.graph.output[0].name
        self._nodes = []
        available = set(self._weight_names) | {self._input_name}
        for node in model.graph.node:
            attributes = {attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute}
            if not all(name in available for name in node.input):
                raise ValueError("ArcFace graph contains an unresolved input")
            if node.op_type == "Flatten" and attributes.get("axis", 1) != 1:
                raise ValueError("ArcFace requires Flatten axis=1")
            if node.op_type == "Conv":
                pads = attributes.get("pads", [0, 0, 0, 0])
                if len(pads) != 4 or pads[:2] != pads[2:]:
                    raise ValueError("ArcFace requires symmetric 2D convolution padding")
            self._nodes.append((node.op_type, tuple(node.input), node.output[0], attributes))
            available.add(node.output[0])
        if self._output_name not in available:
            raise ValueError("ArcFace graph embedding output is missing")
        self._uses = Counter(
            name for _, inputs, _, _ in self._nodes for name in inputs if name not in self._weight_names
        )
        self.requires_grad_(False).eval()

    def train(self, mode=True):
        return super().train(False)

    def forward(self, images):
        with torch.amp.autocast(device_type=images.device.type, enabled=False):
            values = {self._input_name: images.float()}
            uses = self._uses.copy()
            for op, inputs, output, attrs in self._nodes:
                args = [values[name] if name in values else getattr(self, self._weight_names[name])
                        for name in inputs]
                if op == "Conv":
                    result = F.conv2d(
                        args[0], args[1], args[2] if len(args) == 3 else None,
                        stride=attrs.get("strides", 1),
                        padding=attrs.get("pads", [0, 0, 0, 0])[:2],
                        dilation=attrs.get("dilations", 1), groups=attrs.get("group", 1),
                    )
                elif op == "PRelu":
                    result = F.prelu(args[0], args[1].reshape(-1))
                elif op == "BatchNormalization":
                    result = F.batch_norm(
                        args[0], args[3], args[4], args[1], args[2], training=False,
                        eps=attrs.get("epsilon", 1e-5),
                    )
                elif op == "Add":
                    result = args[0] + args[1]
                elif op == "Flatten":
                    result = args[0].flatten(1)
                elif op == "Gemm":
                    left = args[0].T if attrs.get("transA", 0) else args[0]
                    right = args[1].T if attrs.get("transB", 0) else args[1]
                    result = attrs.get("alpha", 1.) * torch.mm(left, right)
                    if len(args) == 3:
                        result = result + attrs.get("beta", 1.) * args[2]
                else:
                    raise ValueError(f"Unsupported ArcFace ONNX operator: {op}")
                values[output] = result
                for name in inputs:
                    if name in uses:
                        uses[name] -= 1
                        if uses[name] == 0:
                            del values[name]
            return values[self._output_name]


def align_arcface_crops(images, affines):
    """Apply input-pixel to 112-pixel affines with gradients only to RGB images.

    ``cv2.warpAffine`` uses black RGB padding and pixel-center coordinates.
    Normalize its inverse into the ``align_corners=False`` sampling convention.
    """
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("ArcFace images must have shape [batch, 3, height, width]")
    matrices = torch.as_tensor(affines, device=images.device, dtype=torch.float32).detach()
    if matrices.shape != (len(images), 2, 3):
        raise ValueError("ArcFace alignment must have shape [batch, 2, 3]")
    if not torch.isfinite(matrices).all() or (torch.linalg.det(matrices[:, :, :2]).abs() < 1e-8).any():
        raise ValueError("ArcFace alignment must be finite and invertible")
    with torch.amp.autocast(device_type=images.device.type, enabled=False):
        batch = len(images)
        height, width = images.shape[-2:]
        homogeneous = torch.eye(3, device=images.device).expand(batch, -1, -1).clone()
        homogeneous[:, :2] = matrices
        input_normalization = images.new_tensor([
            [2 / width, 0, 1 / width - 1],
            [0, 2 / height, 1 / height - 1], [0, 0, 1],
        ], dtype=torch.float32)
        output_pixels = images.new_tensor([[56, 0, 55.5], [0, 56, 55.5], [0, 0, 1]],
                                         dtype=torch.float32)
        theta = (input_normalization @ torch.linalg.inv(homogeneous) @ output_pixels)[:, :2]
        grid = F.affine_grid(theta, (batch, 3, 112, 112), align_corners=False)
        rgb = (images.float().clamp(-1, 1) + 1) * .5
        return F.grid_sample(rgb, grid, mode="bilinear", padding_mode="zeros",
                             align_corners=False) * 2 - 1


class ArcFaceIdentityLoss(nn.Module):
    """Source identity cosine distance with an explicit training crop contract.

    Detection and five-point estimation use detached pixels. The generated RGB
    crop and frozen encoder keep their autograd path. ``paired_target`` uses the
    fixed target portrait's geometry for the generated crop, avoiding changes to
    the training objective from redetecting generated pixels. Target embeddings
    never enter the identity reference. The default ``generated`` keeps dynamic
    generated-face detection. Failed training crop detection uses the source
    affine; evaluation embeddings always require their own successful detection.
    Deep3D alignment arguments are accepted only for compatibility with the
    previous loss and are not ArcFace five-point maps.
    """

    metric_name = "arcface"
    supports_exposure_normalization = False

    def __init__(self, weights_path, processor, *, training_alignment="generated"):
        super().__init__()
        if training_alignment not in ("generated", "paired_target"):
            raise ValueError("ArcFace training_alignment must be 'generated' or 'paired_target'")
        self.encoder = FrozenArcFaceEncoder(weights_path)
        self.processor = processor
        self.training_alignment = training_alignment
        self.last_detection_fallbacks = 0
        self.requires_grad_(False).eval()

    def train(self, mode=True):
        return super().train(False)

    @torch.no_grad()
    def _detect_affines(self, images):
        affines = []
        for image in images.detach():
            rgb = ((image.float().clamp(-1, 1) + 1) * 127.5).round().byte()
            rgb = rgb.permute(1, 2, 0).cpu().numpy()
            face = self.processor.detect_yunet_face(rgb)
            affine = None
            if face is not None:
                points = np.asarray(face[4:14], dtype=np.float32).reshape(5, 2)
                if np.isfinite(points).all():
                    affine, _ = cv2.estimateAffinePartial2D(points, ARCFACE_LANDMARKS, method=cv2.LMEDS)
                if affine is not None and (not np.isfinite(affine).all()
                                           or abs(np.linalg.det(affine[:, :2])) < 1e-8):
                    affine = None
            affines.append(affine)
        return affines

    def _embed_aligned(self, images, affines):
        if any(affine is None for affine in affines):
            raise FaceDetectionError("ArcFace face detection failed; no valid five-point alignment")
        crops = align_arcface_crops(images, np.stack(affines))
        features = self.encoder(crops)
        if not torch.isfinite(features).all() or (features.norm(dim=-1) < 1e-8).any():
            raise ValueError("ArcFace produced invalid identity embeddings")
        return F.normalize(features, dim=-1)

    def embed(self, images, alignment=None, *, normalize_exposure=False):
        if normalize_exposure:
            raise ValueError("ArcFace does not support exposure normalization")
        with torch.amp.autocast(device_type=images.device.type, enabled=False):
            return self._embed_aligned(images.float(), self._detect_affines(images))

    def forward(self, generated, source, target, source_alignment, target_alignment):
        self.last_detection_fallbacks = 0
        with torch.amp.autocast(device_type=generated.device.type, enabled=False):
            with torch.no_grad():
                source_affines = self._detect_affines(source)
                source_embedding = self._embed_aligned(source.float(), source_affines)
            alignment_images = target if self.training_alignment == "paired_target" else generated
            generated_affines = (self._detect_affines(alignment_images)
                                 if alignment_images is not None else [None] * len(generated))
            for index, affine in enumerate(generated_affines):
                if affine is None:
                    generated_affines[index] = source_affines[index]
                    self.last_detection_fallbacks += 1
            embedding = self._embed_aligned(generated.float(), generated_affines)
            return (1 - F.cosine_similarity(embedding, source_embedding)).mean()
