"""Evaluate a trained TcADiff checkpoint on paired holdout portraits."""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import random

import torch
from torchvision.utils import save_image
from tqdm import tqdm

from configs.stage2_config import Stage2Config
from data.datasets import create_dataset
from scripts.inference import _load_config_from_yaml, load_generation_checkpoint
from utils.editing_metrics import MIN_YAW_CHANGE_DEG, SFaceEvaluator, angle_delta_deg, pose_scores, summarize
from utils.metrics import ImageQualityMetrics


def record_pose(row, prefix, pose, target, source, text):
    if pose is None:
        row[prefix + "pose_within_10deg"] = 0
        if abs(angle_delta_deg(float(target[1]), float(source[1]))) >= MIN_YAW_CHANGE_DEG:
            row[prefix + "pose_direction_hit"] = 0
        if any(f"{side} side of the image" in text.lower() for side in ("left", "right")):
            row[prefix + "pose_target_side_hit"] = 0
        return
    row.update({prefix + key: value for key, value in pose_scores(pose, target, source, text).items()})


def record_coefficients(row, model, source, target, predicted, generated, attributes, text):
    groups = model.text_3d_editor.attribute_slices
    source_pose = torch.rad2deg(source[groups["pose"]])
    target_pose = torch.rad2deg(target[groups["pose"]])
    if "pose" in attributes:
        row["target_yaw_deg"] = float(target_pose[1])
        for prefix, coeff in (("source_", source), ("editor_", predicted), ("generated_", generated)):
            pose = torch.rad2deg(coeff[groups["pose"]]) if coeff is not None else None
            record_pose(row, prefix, pose, target_pose, source_pose, text)

    for attribute, group in groups.items():
        if attribute in attributes:
            if attribute == "pose":
                continue
            row["source_" + attribute + "_coeff_mae"] = float((source[group] - target[group]).abs().mean())
            row["editor_" + attribute + "_coeff_mae"] = float((predicted[group] - target[group]).abs().mean())
            if generated is not None:
                row["generated_" + attribute + "_coeff_mae"] = float((generated[group] - target[group]).abs().mean())
        elif generated is not None:
            change = generated[group] - source[group]
            if attribute == "pose":
                change = angle_delta_deg(torch.rad2deg(change), 0)
                row["unrequested_pose_mae_deg"] = float(change.abs().mean())
            else:
                row["unrequested_" + attribute + "_coeff_mae"] = float(change.abs().mean())


@torch.inference_mode()
def evaluate(args, config):
    if args.lpips and not Path(torch.hub.get_dir(), "checkpoints", "alexnet-owt-7be5be79.pth").is_file():
        raise FileNotFoundError("LPIPS requires torchvision AlexNet weights in torch.hub.get_dir()/checkpoints")
    config.lpips_enabled = args.lpips
    config.fid_enabled = False
    # Restore the checkpoint's target coordinate convention before reading pairs.
    model, step = load_generation_checkpoint(config, args.checkpoint, use_default_guidance=not args.config)
    dataset = create_dataset(config, "val")
    indices = [i for i, (_, _, text) in enumerate(dataset.metadata)
               if not args.instruction or args.instruction.lower() in text.lower()]
    if args.limit and args.limit < len(indices):
        indices = sorted(random.Random(args.seed).sample(indices, args.limit))
    if not indices:
        raise ValueError("No validation pairs match the requested instruction")

    if args.guidance is None:
        args.guidance = config.guidance_scale
    if args.strength is None:
        args.strength = config.inference_strength
    if args.steps is None:
        args.steps = config.num_inference_steps
    metrics = ImageQualityMetrics(config, config.device)
    if args.lpips and not metrics.lpips_enabled:
        raise RuntimeError("Install lpips==0.1.4 in the project environment to use --lpips")
    identity = SFaceEvaluator(model.deep3d_processor, args.sface) if args.sface else None

    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    by_instruction = defaultdict(list)
    by_attribute = defaultdict(list)
    identity_features = {}
    saved = set()
    for index in tqdm(indices, desc=f"Evaluating {args.condition}"):
        item = dataset[index]
        source_name, target_name, text = dataset.metadata[index]
        source = item["image"].unsqueeze(0).to(config.device)
        target = item["target_image"].unsqueeze(0).to(config.device)
        theta = item["params"].unsqueeze(0).to(config.device)
        target_theta = item["target_params"].unsqueeze(0).to(config.device)
        attributes = model.text_3d_editor.parse_attributes(text)
        predicted = model.predict_delta(theta, [text], item["rendered_image"].unsqueeze(0).to(config.device))
        delta = predicted
        if args.condition == "oracle":
            mask = model.text_3d_editor.attribute_mask([attributes], theta)
            delta = (target_theta - theta).masked_fill(~mask, 0)
        elif args.condition == "text-only":
            delta = torch.zeros_like(predicted)
        generator = torch.Generator(device=config.device).manual_seed(args.seed + index)
        generated, _ = model.generate(
            source, [text], guidance_scale=args.guidance, num_inference_steps=args.steps,
            generator=generator, strength=args.strength, theta_0=theta, delta_theta=delta,
            estimate_final_theta=False, use_geometry=args.condition != "text-only",
            source_alignment=item["source_alignment"].unsqueeze(0).to(config.device),
        )
        try:
            generated_theta = model.get_3d_params(generated)[0].cpu()
        except ValueError as error:
            if "No facial landmarks detected" not in str(error):
                raise
            generated_theta = None
        row = {"source": source_name, "target": target_name, "instruction": text,
               "attributes": "+".join(attributes), "generated_face_detected": int(generated_theta is not None)}
        record_coefficients(row, model, item["params"], item["target_params"],
                            (theta + predicted)[0].cpu(), generated_theta, attributes, text)
        for prefix, image in (("generated_", generated), ("source_copy_", source)):
            row.update({prefix + key: value for key, value in
                        metrics.compute_per_sample_metrics(target, image).items()})
        if identity:
            for name, image in ((source_name, source), (target_name, target)):
                if name not in identity_features:
                    identity_features[name] = identity.encode(image)
            source_feature = identity_features[source_name]
            generated_feature = identity.encode(generated)
            row["generated_sface_cosine"] = identity.similarity(source_feature, generated_feature)
            row["target_sface_cosine"] = identity.similarity(source_feature, identity_features[target_name])
            if row["generated_sface_cosine"] is not None:
                row["target_matched_sface_cosine"] = row["target_sface_cosine"]

        if text not in saved and len(saved) < args.examples:
            filename = f"example_{len(saved) + 1:02d}.png"
            save_image(torch.cat((source, target, generated)).add(1).div(2).clamp(0, 1),
                       args.output / filename, nrow=3)
            row["example"] = filename
            saved.add(text)
        rows.append(row)
        by_instruction[text].append(row)
        for attribute in attributes:
            by_attribute[attribute].append(row)

    with (args.output / "pairs.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "checkpoint": str(args.checkpoint), "checkpoint_step": step, "pairs": len(rows),
        "validation_pairs_available": len(dataset), "seed": args.seed, "sampling_steps": args.steps,
        "strength": args.strength, "guidance": args.guidance, "condition": args.condition,
        "sface_weights": str(args.sface) if args.sface else None, "lpips": args.lpips,
        "overall": summarize(rows),
        "by_attribute": {name: summarize(group) for name, group in by_attribute.items()},
        "by_instruction": {name: summarize(group) for name, group in by_instruction.items()},
    }
    with (args.output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
    print(f"Evaluation results for {len(rows)} pairs saved in {args.output}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config")
    parser.add_argument("--data_root")
    parser.add_argument("--clip_local_path")
    parser.add_argument("--diffusion_local_path")
    parser.add_argument("--sface", type=Path, help="Path to local SFace ONNX weights")
    parser.add_argument("--lpips", action="store_true")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--guidance", type=float)
    parser.add_argument("--strength", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--instruction", help="Evaluate only instructions containing this phrase")
    parser.add_argument("--examples", type=int, default=12)
    parser.add_argument("--condition", default="predicted", choices=("predicted", "oracle", "text-only"))
    args = parser.parse_args(argv)
    config = _load_config_from_yaml(args.config, Stage2Config) if args.config else Stage2Config()
    for field in ("data_root", "clip_local_path", "diffusion_local_path"):
        if getattr(args, field):
            setattr(config, field, getattr(args, field))
    if (args.steps is not None and args.steps < 1
            or args.limit is not None and args.limit < 1
            or args.strength is not None and not 0 < args.strength <= 1):
        parser.error("Expected positive steps and limit, and strength in (0, 1]")
    evaluate(args, config)


if __name__ == "__main__":
    main()
