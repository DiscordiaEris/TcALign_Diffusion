"""Bounded real-model check before launching the repaired server run."""

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from configs.stage1_config import Stage1Config
from configs.stage2_config import Stage2Config
from data.datasets import create_dataset
from models.tcalign_diffusion import TcAlignDiffusion
from scripts.inference import _load_config_from_yaml
from trainers.stage1_trainer import Stage1Trainer
from trainers.stage2_trainer import Stage2Trainer


def run():
    torch.manual_seed(42)
    output = Path("outputs/repair_diagnostics_20261120/smoke")
    output.mkdir(parents=True, exist_ok=True)
    stage1 = _load_config_from_yaml("configs/runs/fei_stage1_repair.yaml", Stage1Config)
    stage2 = _load_config_from_yaml("configs/runs/fei_stage2_repair.yaml", Stage2Config)
    for config in (stage1, stage2):
        config.num_workers = 0
        config.log_dir = str(output)
        config.output_dir = str(output)
        config.checkpoint_dir = str(output)
        config.log_interval = 1
        config.save_interval = 0
        config.num_epochs = 1
        config.num_inference_steps = 15
    train = create_dataset(stage2, "train")
    validation = create_dataset(stage2, "val")
    # One pair per canonical instruction, spanning all attribute groups.
    train.metadata = list({text: (source, target, text)
                           for source, target, text in reversed(train.metadata)}.values())
    validation.metadata = list({text: (source, target, text)
                                for source, target, text in reversed(validation.metadata)}.values())
    model = TcAlignDiffusion(stage2)
    first = Stage1Trainer(stage1, model, DataLoader(train, batch_size=2))
    first_loss = first.train_one_epoch()
    assert torch.isfinite(torch.tensor(first_loss))
    first_checkpoint = str(output / "stage1.pth")
    first.save_checkpoint(first_checkpoint)
    stage2.stage1_checkpoint_path = first_checkpoint
    del first
    second = Stage2Trainer(stage2, model, DataLoader(train, batch_size=1),
                           DataLoader(validation, batch_size=1))
    key = "down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_k.weight"
    attention = dict(model.diffusion_unet.unet.named_parameters())[key]
    original_norm = float(attention.detach().norm())
    second_loss = second.train_one_epoch()
    assert torch.isfinite(torch.tensor(second_loss))
    checkpoint = str(output / "stage2.pth")
    second.save_checkpoint(checkpoint)
    second.load_checkpoint(checkpoint)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert saved["inference_config"]["inference_strength"] == 0.65
    assert saved["training_config"]["measured_geometry_probability"] == 0
    del saved
    first_val = second.evaluate()
    second_val = second.evaluate()
    assert first_val == second_val, (first_val, second_val)
    ratio = float(attention.detach().norm()) / original_norm
    assert 0.95 < ratio < 1.05, ratio
    report = {"stage1_steps": len(DataLoader(train, batch_size=2)),
              "stage2_steps": second.global_step, "stage1_loss": first_loss,
              "stage2_loss": second_loss, "validation_loss": first_val,
              "attention_norm_retention": ratio,
              "repeat_validation_identical": True, "checkpoint_reload": True}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    run()
