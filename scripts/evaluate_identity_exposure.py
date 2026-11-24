"""Fixed-noise held-out identity/exposure validation of the final generated images."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torchvision.utils import save_image

from configs.stage2_config import Stage2Config
from data.datasets import FFHQDataset
from data.transforms import create_transforms, create_rendered_transform
from models.photometric_exposure import aligned_face_luminance
from scripts.inference import _load_config_from_yaml, load_generation_checkpoint
from utils.editing_metrics import SFaceEvaluator, summarize
from utils.losses import requested_lighting_exposure_loss


TASKS = (
    ('left', 'Turn the head toward the left side of the image.', 'Turn the head toward the left side of the image.'),
    ('right', 'Turn the head slightly toward the right side of the image.', 'Turn the head slightly toward the right side of the image.'),
    ('smile', 'Make the person smile.', 'Make the person smile.'),
    ('dim', 'Use dim lighting on the face.', 'Dim the lighting on the face.'),
    ('very_dim', 'Use very dim lighting on the face.', 'Make the lighting on the face very dim.'),
)


def evaluate(config_path, checkpoint_path, output_dir, subjects=6, baseline_dirs=()):
    output_dir.mkdir(parents=True, exist_ok=True)
    config = _load_config_from_yaml(config_path, Stage2Config)
    dataset = FFHQDataset(config.data_root, transform=create_transforms(config.image_size, 'val'),
        split='val', config=config, rendered_transform=create_rendered_transform(config.render_image_size), precompute_3d=False)
    groups = {}
    for index, (source, _, text) in enumerate(dataset.metadata):
        if source.endswith('-11.jpg'):
            groups.setdefault(source, {})[text] = index
    chosen = [(source, mapping) for source, mapping in groups.items()
              if all(t[1] in mapping for t in TASKS)
              and (not baseline_dirs or any((directory / f'before_{source}.png').is_file()
                                            for directory in baseline_dirs))][:subjects]
    if len(chosen) != subjects:
        raise ValueError(f'Expected {subjects} complete held-out identities, found {len(chosen)}')
    model, step = load_generation_checkpoint(config, checkpoint_path)
    model.initialize_identity_loss()
    sface = SFaceEvaluator(model.deep3d_processor, 'models/face_recognition_sface_2021dec.onnx')
    rows = []

    def baseline(source, column):
        for directory in baseline_dirs:
            path = directory / f'before_{source}.png'
            if path.is_file():
                pixels = np.array(Image.open(path).convert('RGB'))
                h = (pixels.shape[0] - 8) // 3
                w = (pixels.shape[1] - 12) // 5
                crop = pixels[2 + 2 * (h + 2):2 + 2 * (h + 2) + h,
                              2 + column * (w + 2):2 + column * (w + 2) + w].copy()
                return torch.from_numpy(crop).permute(2, 0, 1)[None].to(config.device).float() / 127.5 - 1
        return None

    def exposure_normalized(image, affine):
        mean = aligned_face_luminance(image, affine).clamp_min(.05)
        return (((image + 1) * .5) * (.5 / mean)[:, None, None, None]).clamp(0, 1) * 2 - 1

    with torch.inference_mode():
        for source, mapping in chosen:
            images = []
            for column, (task, target_text, request_text) in enumerate(TASKS):
                batch = dataset[mapping[target_text]]
                x = batch['image'][None].to(config.device)
                y = batch['target_image'][None].to(config.device)
                affine = batch['source_alignment'][None].to(config.device)
                generated, _ = model.generate(x, [request_text], theta_0=batch['params'][None].to(config.device),
                    source_alignment=affine, source_rendered=batch['rendered_image'][None].to(config.device),
                    estimate_final_theta=False, guidance_scale=config.guidance_scale,
                    num_inference_steps=config.num_inference_steps, strength=config.inference_strength,
                    generator=torch.Generator(device=config.device).manual_seed(config.seed))
                old = baseline(source, column)
                candidates = [('after', generated)] if old is None else [('before', old), ('after', generated)]
                features = model.identity_loss
                source_raw = features.embed(x, affine)
                source_normal = features.embed(x, affine, normalize_exposure=True)
                source_sface = sface.encode(x)
                source_sface_normal = sface.encode(exposure_normalized(x, affine))
                target_coeff = model.deep3d_processor.reconstruct_aligned(y, affine)
                for label, image in candidates:
                    identity = float(F.cosine_similarity(source_raw, features.embed(image, affine)))
                    identity_normal = float(F.cosine_similarity(source_normal, features.embed(image, affine, normalize_exposure=True)))
                    generated_coeff = model.deep3d_processor.reconstruct_aligned(image, affine)
                    row = dict(label=label, source=source, task=task, text=request_text,
                        identity=identity, identity_exposure_normalized=identity_normal,
                        sface=sface.similarity(source_sface, sface.encode(image)),
                        sface_exposure_normalized=sface.similarity(source_sface_normal, sface.encode(exposure_normalized(image, affine))),
                        yaw=float(generated_coeff[:, 145].rad2deg()), target_yaw=float(target_coeff[:, 145].rad2deg()),
                        source_luminance=float(aligned_face_luminance(x, affine)),
                        target_luminance=float(aligned_face_luminance(y, affine)),
                        generated_luminance=float(aligned_face_luminance(image, affine)))
                    if task in ('left', 'right'):
                        row['yaw_error_deg'] = abs(row['yaw'] - row['target_yaw'])
                    if task in ('dim', 'very_dim'):
                        row['exposure_error'] = float(requested_lighting_exposure_loss(image, x, y, affine, [['lighting']]))
                    rows.append(row)
                    print(json.dumps(row), flush=True)
                images.append((x, y, old if old is not None else x, generated))
            save_image(torch.cat([triple[j] for j in range(4) for triple in images]).add(1).div(2),
                       output_dir / f'{source}.png', nrow=5)
            (output_dir / 'rows.json').write_text(json.dumps(rows, indent=2) + '\n')
    report = {'checkpoint': str(checkpoint_path), 'step': step, 'subjects': subjects, 'seed': config.seed,
              'guidance': config.guidance_scale, 'strength': config.inference_strength,
              'summary': {label: {task: summarize([r for r in rows if r['label'] == label and r['task'] == task])
                                    for task, _, _ in TASKS} for label in ('before', 'after')}}
    (output_dir / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--subjects', type=int, default=6)
    parser.add_argument('--baseline-dir', type=Path, action='append', default=[])
    args = parser.parse_args()
    evaluate(args.config, args.checkpoint, args.output, args.subjects, args.baseline_dir)


if __name__ == '__main__':
    main()
