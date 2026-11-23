import argparse
import os
import torch
import yaml
import random
import numpy as np
from configs.stage2_config import Stage2Config
from data.datasets import create_dataset, create_training_sampler
from models.tcalign_diffusion import TcAlignDiffusion
from trainers.stage2_trainer import Stage2Trainer
from torch.utils.data import DataLoader


def _load_config_from_yaml(yaml_path, config_cls):
    """从 YAML 文件加载配置并覆盖默认值"""
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    valid_keys = {f.name for f in config_cls.__dataclass_fields__.values()}
    filtered = {k: v for k, v in data.items() if k in valid_keys}
    return config_cls(**filtered)


def main(argv=None):
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Train Stage 2: Conditional Diffusion Mapping")
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--data_root", type=str, help="Path to FFHQ dataset root")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--init_checkpoint", type=str, help="Initialize model weights only; start a new objective/optimizer run")
    parser.add_argument("--stage1_ckpt", type=str, required=True, help="Path to stage1 checkpoint")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--num_epochs", type=int, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--clip_local_path", type=str, help="Local path to CLIP model folder")
    parser.add_argument("--diffusion_local_path", type=str, help="Local path to Stable Diffusion model folder")
    args = parser.parse_args(argv)
    if args.resume and args.init_checkpoint:
        parser.error("Choose either resume or init_checkpoint")

    # 加载配置: YAML 文件 → CLI 参数覆盖
    if args.config:
        if not os.path.isfile(args.config):
            raise FileNotFoundError(f"Config file not found: {args.config}")
        config = _load_config_from_yaml(args.config, Stage2Config)
    else:
        config = Stage2Config()

    # CLI 参数覆盖配置
    if args.data_root:
        config.data_root = args.data_root
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.num_epochs:
        config.num_epochs = args.num_epochs
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.clip_local_path:
        config.clip_local_path = args.clip_local_path
    if args.diffusion_local_path:
        config.diffusion_local_path = args.diffusion_local_path

    config.stage1_checkpoint_path = args.stage1_ckpt
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)

    # 创建数据加载器
    train_dataset = create_dataset(config, "train")
    val_dataset = create_dataset(config, "val")
    train_sampler = create_training_sampler(train_dataset, config)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=config.num_workers
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers
    )

    # 创建模型
    model = TcAlignDiffusion(config)

    # 创建训练器
    trainer = Stage2Trainer(config, model, train_dataloader, val_dataloader)

    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location="cpu", weights_only=True)
        model.load_checkpoint_state(checkpoint["model_state_dict"])
        trainer.logger.info("Initialized weights from %s at step %s; new optimizer/objective",
                            args.init_checkpoint, checkpoint.get("global_step"))
        del checkpoint

    # 恢复训练（如果指定）
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # 开始训练
    trainer.train()


if __name__ == "__main__":
    main()
