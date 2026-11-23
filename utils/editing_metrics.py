"""Per-example edit metrics and aggregation with explicit detection coverage."""

from collections import defaultdict
from statistics import mean

import cv2
import numpy as np


MIN_YAW_CHANGE_DEG = 5.0
TARGET_YAW_TOLERANCE_DEG = 10.0


def coefficient_edit_scores(source, target, generated, attributes, groups):
    """Separate requested edit errors from drift; include the no-edit baseline.

    Uses a fixed aligned Deep3D estimator, so these are monitoring proxies,
    not face-detection success or independent perceptual quality scores.
    """
    values = defaultdict(list)
    for index, requested in enumerate(attributes):
        for name, group in groups.items():
            if name in requested:
                for prefix, coeff in (("generated", generated), ("source_copy", source)):
                    error = coeff[index, group] - target[index, group]
                    if name == "pose":
                        error = angle_delta_deg(error.rad2deg(), 0)
                    suffix = "mae_deg" if name == "pose" else "coeff_mae"
                    values[f"{prefix}_{name}_{suffix}"].append(float(error.abs().mean()))
            else:
                drift = generated[index, group] - source[index, group]
                if name == "pose":
                    drift = angle_delta_deg(drift.rad2deg(), 0)
                values[f"unrequested_{name}_drift"].append(float(drift.abs().mean()))
    return {name: mean(items) for name, items in values.items()}


def angle_delta_deg(angle, reference):
    return (angle - reference + 180) % 360 - 180


def pose_scores(pose, target, source, prompt):
    errors = angle_delta_deg(pose, target).abs()
    yaw, target_yaw, source_yaw = map(float, (pose[1], target[1], source[1]))
    desired = angle_delta_deg(target_yaw, source_yaw)
    moved = angle_delta_deg(yaw, source_yaw)
    scores = {
        "yaw_deg": yaw,
        "yaw_abs_error_deg": float(errors[1]),
        "pose_mae_deg": float(errors.mean()),
        "pose_within_10deg": int(errors[1] <= TARGET_YAW_TOLERANCE_DEG),
    }
    if abs(desired) >= MIN_YAW_CHANGE_DEG:
        scores["pose_direction_hit"] = int(moved * (1 if desired > 0 else -1) >= MIN_YAW_CHANGE_DEG)
    # The FEI prompts refer to image coordinates, where negative yaw in the
    # project's Deep3D model points to the viewer's left.
    if "left side of the image" in prompt.lower():
        scores["pose_target_side_hit"] = int(yaw <= -MIN_YAW_CHANGE_DEG)
    if "right side of the image" in prompt.lower():
        scores["pose_target_side_hit"] = int(yaw >= MIN_YAW_CHANGE_DEG)
    return scores


def summarize(rows):
    values = defaultdict(list)
    for row in rows:
        for key, value in row.items():
            if isinstance(value, (float, int)) and np.isfinite(value):
                values[key].append(value)
    return {key: {"mean": mean(items), "n": len(items)} for key, items in sorted(values.items())}


class SFaceEvaluator:
    """Compare aligned identity embeddings without using the editor's CLIP model."""

    def __init__(self, processor, weights):
        self.processor = processor
        self.recognizer = cv2.FaceRecognizerSF.create(str(weights), "")

    def encode(self, image):
        pixels = ((image[0].float().clamp(-1, 1) + 1) * 127.5).round().byte()
        rgb = pixels.permute(1, 2, 0).cpu().numpy()
        face = self.processor.detect_yunet_face(rgb)
        if face is None:
            return None
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cropped = self.recognizer.alignCrop(bgr, face)
        return self.recognizer.feature(cropped).copy()

    def similarity(self, source, edited):
        if source is None or edited is None:
            return None
        return float(self.recognizer.match(source, edited, cv2.FaceRecognizerSF_FR_COSINE))
