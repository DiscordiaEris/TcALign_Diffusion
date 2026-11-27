"""Fixed sampled edits for identity-aware checkpoint selection.

The caller owns model mode and the global RNG context. This module generates
one paired example at a time and keeps missing detections in coverage counts.
"""

import math
import re
from collections import OrderedDict
from pathlib import Path
from statistics import mean

import torch
from torch.nn import functional as F

from .editing_metrics import SFaceEvaluator, pose_scores
from .losses import requested_attribute_control_loss, requested_lighting_exposure_loss


# Metadata prompts identify the paired target; exposure generation deliberately
# uses the legacy wording, which differs from the canonical exposure labels.
_PANEL = (
    ("left", "Turn the head toward the left side of the image.",
     "Turn the head toward the left side of the image."),
    ("right", "Turn the head slightly toward the right side of the image.",
     "Turn the head slightly toward the right side of the image."),
    ("smile", "Make the person smile.", "Make the person smile."),
    ("dim", "Use dim lighting on the face.", "Dim the lighting on the face."),
    ("very_dim", "Use very dim lighting on the face.", "Make the lighting on the face very dim."),
)


def _finite(value):
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _average(values):
    return mean(values) if values else None


def _select_pairs(dataset, subjects):
    candidates = OrderedDict()
    canonical = {text for _, text, _ in _PANEL}
    for index, (source, target, text) in enumerate(dataset.metadata):
        match = re.fullmatch(r"(fei-\d+)-11\.[^.]+", Path(source).name, flags=re.IGNORECASE)
        if match and text in canonical:
            identity, pairs = candidates.setdefault(source, (match.group(1).lower(), {}))
            pairs.setdefault(text, (index, target))
    selected, seen = [], set()
    for source, (identity, pairs) in candidates.items():
        if canonical.issubset(pairs) and identity not in seen:
            selected.append((source, pairs))
            seen.add(identity)
            if len(selected) == subjects:
                break
    return selected


@torch.no_grad()
def evaluate_sampled_identity(model, dataset, device, *, subjects=6, seed=42,
                              evaluator=None, sface_weights=None):
    """Return ``rows`` and a flat ``aggregate`` for a fixed frontal-source panel.

    ``evaluator`` follows ``SFaceEvaluator.encode/similarity`` and permits CPU
    tests without recognition weights. Identity selection averages only the
    left/right/smile edits. Lighting edits retain their own detection results.
    Fewer complete subjects are reported as incomplete coverage, never silently
    accepted as a smaller validation panel. Optional FaceNet scores are reported
    in both raw and exposure-normalized form when ``model.identity_loss`` exists.
    Smile expression errors use the training control loss's displacement scale.
    """
    if not isinstance(subjects, int) or isinstance(subjects, bool) or subjects < 1:
        raise ValueError("subjects must be a positive integer")
    device = torch.device(device)
    if evaluator is None:
        weights = (Path(sface_weights) if sface_weights is not None else
                   Path(__file__).resolve().parents[1] / "models/face_recognition_sface_2021dec.onnx")
        evaluator = SFaceEvaluator(model.deep3d_processor, weights)

    selected = _select_pairs(dataset, subjects)
    rows = []
    identity_scores, target_identity_scores, pose_errors, exposures = [], [], [], []
    expression_errors = []
    expression_progresses = []
    expression_direction_hits = 0
    facenet_scores = {"raw": [], "normalized": []}
    direction_hits = 0
    identity = getattr(model, "identity_loss", None)
    identity_name = getattr(identity, "metric_name", "facenet")
    normalizations = (("raw", False), ("normalized", True)) if getattr(
        identity, "supports_exposure_normalization", True) else (("raw", False),)
    detection_errors = ()
    if identity_name == "arcface":
        from models.arcface_identity_loss import FaceDetectionError
        detection_errors = (FaceDetectionError,)
    for source_name, pairs in selected:
        source_feature = None
        source_encoded = False
        source_pose = None
        for edit, canonical, prompt in _PANEL:
            index, target_name = pairs[canonical]
            sample = dataset[index]

            def batch(key):
                return sample[key].unsqueeze(0).to(device)

            source, target = batch("image"), batch("target_image")
            source_alignment, target_alignment = batch("source_alignment"), batch("target_alignment")
            generator = torch.Generator(device=device).manual_seed(seed)
            generated, _ = model.generate(
                source, [prompt], guidance_scale=model.config.guidance_scale,
                num_inference_steps=50, strength=model.config.inference_strength,
                generator=generator, theta_0=batch("params"),
                source_alignment=source_alignment, source_rendered=batch("rendered_image"),
                estimate_final_theta=False,
            )
            if not source_encoded:
                source_feature = evaluator.encode(source)
                source_encoded = True
            generated_feature, target_feature = evaluator.encode(generated), evaluator.encode(target)
            similarity = evaluator.similarity(source_feature, generated_feature)
            target_similarity = evaluator.similarity(source_feature, target_feature)
            valid = _finite(similarity)
            row = {
                "source": source_name, "target": target_name, "edit": edit,
                "canonical_text": canonical, "text": prompt,
                "sface_cosine": float(similarity) if valid else None,
                "target_sface_cosine": float(target_similarity) if _finite(target_similarity) else None,
                "source_face_detected": source_feature is not None,
                "generated_face_detected": generated_feature is not None,
                "target_face_detected": target_feature is not None,
                "identity_valid": valid,
            }
            identity_edit = edit in {"left", "right", "smile"}
            if identity_edit:
                if valid:
                    identity_scores.append(float(similarity))
                if _finite(target_similarity):
                    target_identity_scores.append(float(target_similarity))

            if edit in {"left", "right"}:
                processor = model.deep3d_processor
                group = model.text_3d_editor.attribute_slices["pose"]
                if source_pose is None:
                    source_pose = processor.reconstruct_aligned(source, source_alignment)[0, group].rad2deg()
                generated_pose = processor.reconstruct_aligned(generated, target_alignment)[0, group].rad2deg()
                target_pose = processor.reconstruct_aligned(target, target_alignment)[0, group].rad2deg()
                pose_valid = all(bool(torch.isfinite(pose).all())
                                 for pose in (source_pose, generated_pose, target_pose))
                row["pose_valid"] = pose_valid
                if pose_valid:
                    scores = pose_scores(generated_pose, target_pose, source_pose, canonical)
                    # Require displacement in the measured target direction and
                    # the requested side of the image, both by at least 5 deg.
                    scores["pose_direction_hit"] = int(
                        scores.get("pose_direction_hit", 0) and scores.get("pose_target_side_hit", 0))
                    row.update(scores)
                    pose_errors.append(scores["yaw_abs_error_deg"])
                    direction_hits += scores["pose_direction_hit"]
            if edit == "smile":
                groups = model.text_3d_editor.attribute_slices
                expression_group = groups.get("expression")
                expression_error = None
                expression_progress = None
                displacement_value = None
                if expression_group is not None:
                    processor = model.deep3d_processor
                    source_coefficients = processor.reconstruct_aligned(source, source_alignment)
                    generated_coefficients = processor.reconstruct_aligned(generated, target_alignment)
                    target_coefficients = processor.reconstruct_aligned(target, target_alignment)
                    expression_valid = all(
                        coefficients[:, expression_group].numel() > 0
                        and bool(torch.isfinite(coefficients[:, expression_group]).all())
                        for coefficients in (source_coefficients, generated_coefficients,
                                             target_coefficients)
                    )
                    if expression_valid:
                        displacement = (target_coefficients[:, expression_group].float()
                                        - source_coefficients[:, expression_group].float())
                        displacement_mse = displacement.square().mean()
                        if _finite(float(displacement_mse)):
                            displacement_value = float(displacement_mse)
                        expression_valid = bool(displacement_mse >= 1e-6)
                    if expression_valid:
                        error = float(requested_attribute_control_loss(
                            generated_coefficients, source_coefficients, target_coefficients,
                            [["expression"]], groups,
                        ))
                        movement = (generated_coefficients[:, expression_group].float()
                                    - source_coefficients[:, expression_group].float())
                        progress = float((movement * displacement).mean() / displacement_mse)
                        if _finite(error) and _finite(progress):
                            expression_error = error
                            expression_errors.append(error)
                            expression_progress = progress
                            expression_progresses.append(min(1.0, max(0.0, progress)))
                            expression_direction_hits += int(progress >= 0.1)
                row["expression_valid"] = expression_error is not None
                row["expression_error_ratio"] = expression_error
                row["expression_progress"] = expression_progress
                row["expression_target_displacement_mse"] = displacement_value
                row["expression_direction_hit"] = int(
                    expression_progress is not None and expression_progress >= 0.1)
            if edit in {"dim", "very_dim"}:
                exposure = float(requested_lighting_exposure_loss(
                    generated, source, target, source_alignment, [["lighting"]]))
                row["exposure_error"] = exposure if _finite(exposure) else None
                row["exposure_valid"] = _finite(exposure)
                if _finite(exposure):
                    exposures.append(exposure)
            if identity is not None:
                for label, normalized in normalizations:
                    try:
                        source_embedding = identity.embed(source, source_alignment, normalize_exposure=normalized)
                        generated_embedding = identity.embed(generated, target_alignment, normalize_exposure=normalized)
                        value = float(F.cosine_similarity(source_embedding.float(), generated_embedding.float()).mean())
                    except detection_errors:
                        value = None
                    row[f"{identity_name}_{label}_cosine"] = value if _finite(value) else None
                    if identity_edit and _finite(value):
                        facenet_scores[label].append(value)
            rows.append(row)

    aggregate = {
        "subjects_expected": subjects, "subjects_selected": len(selected),
        "identity_expected": subjects * 3, "identity_valid": len(identity_scores),
        "identity_score": _average(identity_scores),
        "target_identity_valid": len(target_identity_scores),
        "target_identity_score": _average(target_identity_scores),
        "pose_expected": subjects * 2, "pose_valid": len(pose_errors),
        "pose_direction_hits": direction_hits,
        "pose_yaw_abs_error_deg": _average(pose_errors),
        "expression_expected": subjects, "expression_valid": len(expression_errors),
        "expression_error_ratio": _average(expression_errors),
        "expression_progress_valid": len(expression_progresses),
        "expression_direction_hits": expression_direction_hits,
        "expression_progress": _average(expression_progresses),
        "exposure_expected": subjects * 2, "exposure_valid": len(exposures),
        "exposure_error": _average(exposures),
    }
    if identity is not None:
        for label, _ in normalizations:
            values = facenet_scores[label]
            aggregate[f"{identity_name}_{label}_cosine"] = _average(values)
            aggregate[f"{identity_name}_{label}_valid"] = len(values)
    return {"rows": rows, "aggregate": aggregate}


def should_replace_identity_checkpoint(metrics, best_score, *, max_pose_error=8.0,
                                       max_exposure_error=0.1, require_expression=False,
                                       max_expression_error=1.0, min_expression_progress=None,
                                       min_arcface_score=None):
    """Require the full six-subject panel and actual sampled edit competence.

    Accept either the evaluator result or its aggregate. A missing/nonfinite
    score, incomplete coverage or any failed pose direction rejects promotion.
    ``best_score=None`` represents no previous eligible identity checkpoint.
    Opt-in expression control requires all six smiles and a mean error strictly
    below both source-copy error and the requested expression-error limit.
    Supplying a progress floor instead protects a measured baseline: all six
    subjects must move toward the expression target, with bounded mean error
    and mean progress clipped per subject to prevent overshoot compensation.
    """
    metrics = metrics.get("aggregate", metrics)
    required_counts = {
        "identity_expected": 18, "identity_valid": 18,
        "pose_expected": 12, "pose_valid": 12, "pose_direction_hits": 12,
        "exposure_expected": 12, "exposure_valid": 12,
    }
    if any(metrics.get(key) != expected for key, expected in required_counts.items()):
        return False
    if "arcface_raw_valid" in metrics and metrics["arcface_raw_valid"] != 18:
        return False
    if min_arcface_score is not None:
        arcface = metrics.get("arcface_raw_cosine")
        if (metrics.get("arcface_raw_valid") != 18
                or not all(_finite(value) for value in (arcface, min_arcface_score))
                or arcface < min_arcface_score - 1e-4):
            return False
    score = metrics.get("identity_score")
    pose_error, exposure_error = metrics.get("pose_yaw_abs_error_deg"), metrics.get("exposure_error")
    if not all(_finite(value) for value in (score, pose_error, exposure_error,
                                            max_pose_error, max_exposure_error)):
        return False
    if not (0 <= pose_error <= max_pose_error and 0 <= exposure_error <= max_exposure_error):
        return False
    if require_expression or min_expression_progress is not None:
        if metrics.get("expression_expected") != 6 or metrics.get("expression_valid") != 6:
            return False
        expression_error = metrics.get("expression_error_ratio")
        if not all(_finite(value) for value in (expression_error, max_expression_error)):
            return False
        if min_expression_progress is None:
            if not 0 <= expression_error < min(1.0, max_expression_error):
                return False
        else:
            if (metrics.get("expression_progress_valid") != 6
                    or metrics.get("expression_direction_hits") != 6):
                return False
            progress = metrics.get("expression_progress")
            if not all(_finite(value) for value in (progress, min_expression_progress)):
                return False
            if not (0 <= expression_error <= max_expression_error
                    and 0 <= min_expression_progress <= progress <= 1):
                return False
    if best_score is None:
        best_score = -math.inf
    if math.isnan(float(best_score)):
        return False
    return score > best_score + 1e-4
