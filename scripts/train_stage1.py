import argparse
import os
import yaml
from configs.stage1_config import Stage1Config
from data.datasets import create_dataset
from models.tcalign_diffusion import TcAlignDiffusion
from trainers.stage1_trainer import Stage1Trainer
from torch.utils.data import DataLoader


def _load_config_from_yaml(yaml_path, config_cls):
    """从 YAML 文件加载配置并覆盖默认值"""
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    # 只保留 dataclass 中已有的字段
    valid_keys = {f.name for f in config_cls.__dataclass_fields__.values()}
    filtered = {k: v for k, v in data.items() if k in valid_keys}
    return config_cls(**filtered)


def main(argv=None):
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Train Stage 1: Text-Conditional 3D Editor")
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--data_root", type=str, help="Path to FFHQ dataset root")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--num_epochs", type=int, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--clip_local_path", type=str, help="Local path to CLIP model folder")
    args = parser.parse_args(argv)

    # 加载配置: YAML 文件 → CLI 参数覆盖
    if args.config:
        if not os.path.isfile(args.config):
            raise FileNotFoundError(f"Config file not found: {args.config}")
        config = _load_config_from_yaml(args.config, Stage1Config)
    else:
        config = Stage1Config()

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

    # 创建数据加载器
    train_dataset = create_dataset(config, "train")
    val_dataset = create_dataset(config, "val")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
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
    trainer = Stage1Trainer(config, model, train_dataloader, val_dataloader)

    # 恢复训练（如果指定）
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # 开始训练
    trainer.train()


if __name__ == "__main__":
    main()