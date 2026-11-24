"""Fit only the exposure head on controlled training pairs, then create a weights-only warm start."""
import argparse
import copy
import json
import random
from dataclasses import asdict
from pathlib import Path

import torch
from torch.nn import functional as F
from transformers import CLIPTextModel, CLIPTokenizer

from configs.stage2_config import Stage2Config
from data.datasets import INSTRUCTION_PARAPHRASES
from models.photometric_exposure import PhotometricExposureHead
from scripts.inference import _load_config_from_yaml


def bootstrap(config_path, initial_checkpoint, output_path, report_path, steps=2000):
    config = _load_config_from_yaml(config_path, Stage2Config)
    manifest = json.loads((Path(config.data_root) / 'manifest.json').read_text())
    if not manifest.get('complete') or manifest.get('kind') == 'real_relighting':
        raise ValueError('A completed controlled-exposure dataset is required')
    pairs = manifest['pairs']
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    variants = {}
    legacy = {'dim': 'Dim the lighting on the face.',
              'very_dim': 'Make the lighting on the face very dim.',
              'bright': 'Use bright, even lighting on the face.'}
    for pair in pairs:
        text, state = pair['text'], pair['state']
        if text in variants:
            continue
        candidates = INSTRUCTION_PARAPHRASES.get(legacy[state], ())
        variants[text] = list(dict.fromkeys([text, *candidates]))
    tokenizer = CLIPTokenizer.from_pretrained(config.diffusion_local_path, subfolder='tokenizer', local_files_only=True)
    encoder = CLIPTextModel.from_pretrained(config.diffusion_local_path, subfolder='text_encoder', local_files_only=True).eval()
    text_features = {}
    with torch.no_grad():
        for text in set(legacy.values()) | {text for items in variants.values() for text in items}:
            ids = tokenizer([text], padding='max_length', max_length=77, truncation=True, return_tensors='pt').input_ids
            text_features[text] = encoder(ids).last_hidden_state.mean(dim=1)[0]
    del encoder
    train_ids = sorted({p['identity'] for p in pairs if p['split'] == 'train'})
    random.shuffle(train_ids)
    inner_ids = set(train_ids[:max(1, len(train_ids) // 10)])
    train = [p for p in pairs if p['split'] == 'train' and p['identity'] not in inner_ids]
    inner = [p for p in pairs if p['split'] == 'train' and p['identity'] in inner_ids]
    validation = [p for p in pairs if p['split'] == 'val']
    grouped = {state: [p for p in train if p['state'] == state] for state in legacy}
    head = PhotometricExposureHead(config.condition_embed_dim)
    optimizer = torch.optim.AdamW(head.parameters(), lr=.002, weight_decay=.01)

    def tensors(rows, paraphrase=False, heldout=False):
        texts = [legacy[p['state']] if heldout else random.choice(variants[p['text']]) if paraphrase else p['text'] for p in rows]
        tokens = torch.stack([text_features[t] for t in texts])[:, None]
        source = torch.tensor([p['source_face_luminance'] for p in rows]).clamp_min(1e-4)
        target = torch.tensor([p['actual_target_face_luminance'] for p in rows]).clamp_min(1e-4)
        return tokens, source, target

    best_loss, best_state, best_step = float('inf'), None, 0
    inner_tensors = tensors(inner)
    for step in range(1, steps + 1):
        selected = [random.choice(grouped[random.choice(list(grouped))]) for _ in range(96)]
        tokens, source, target = tensors(selected, paraphrase=True)
        predicted = head(tokens, source.log())
        loss = F.smooth_l1_loss(predicted, target.log(), beta=.1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            with torch.no_grad():
                t, s, y = inner_tensors
                value = float(F.mse_loss(head(t, s.log()), y.log()))
            if value < best_loss:
                best_loss, best_state, best_step = value, copy.deepcopy(head.state_dict()), step
    head.load_state_dict(best_state)
    head.eval()
    report = {'inner_holdout_identities': sorted(inner_ids), 'inner_best_log_mse': best_loss,
              'best_step': best_step, 'training_identities': len(train_ids) - len(inner_ids), 'validation': []}
    for heldout in (False, True):
        tokens, source, target = tensors(validation, heldout=heldout)
        with torch.no_grad():
            prediction = head(tokens, source.log()).exp()
        errors = (prediction - target).square() / (target - source).square().clamp_min(.025 ** 2)
        report['validation'].append({'heldout_wording': heldout, 'n': len(validation),
                                      'normalized_exposure_error': float(errors.mean()),
                                      'mean_absolute_rgb_luminance_error': float((prediction - target).abs().mean())})
        if heldout:
            report['normal_source_rows'] = [dict(identity=p['identity'], state=p['state'], source=p['source'],
                source_luminance=float(s), target_luminance=float(y), predicted_luminance=float(z),
                normalized_error=float(error)) for p, s, y, z, error in zip(validation, source, target, prediction, errors)
                if p['source'].endswith('-11.jpg')]
    original = torch.load(initial_checkpoint, map_location='cpu', weights_only=True)
    state = original['model_state_dict']
    state.update({'photometric_head.' + name: tensor for name, tensor in head.state_dict().items()})
    inference_names = ('guidance_scale', 'num_inference_steps', 'inference_strength', 'use_bf16',
                       'faceid_adapter_path', 'faceid_encoder_path', 'faceid_scale', 'use_photometric_head')
    checkpoint = dict(model_state_dict=state, global_step=0, epoch=0,
        source_checkpoint=str(initial_checkpoint), source_global_step=original['global_step'],
        inference_config={name: getattr(config, name) for name in inference_names}, training_config=asdict(config))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(output_path)
    torch.save(checkpoint, output_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'normal_source_rows'}, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--initial', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=2000)
    args = parser.parse_args()
    bootstrap(args.config, args.initial, args.output, args.report, args.steps)


if __name__ == '__main__':
    main()
