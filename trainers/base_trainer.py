import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.cuda.amp import autocast, GradScaler
import logging
import os
from tqdm import tqdm
from datetime import datetime


class BaseTrainer:
    """基础训练器"""

    def __init__(self, config, model, train_dataloader, val_dataloader=None):
        self.config = config
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        # 设备设置
        self.device = config.device
        self.model.to(self.device)

        # 优化器
        self.optimizer = self.configure_optimizers()

        # 混合精度训练
        self.scaler = GradScaler()

        # 设置日志
        self.setup_logging()

        # 训练状态
        self.global_step = 0
        self.epoch = 0

    def configure_optimizers(self):
        """配置优化器"""
        return Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2),
            weight_decay=self.config.weight_decay
        )

    def setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(
                    os.path.join(self.config.log_dir, f'train_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def save_checkpoint(self, checkpoint_path):
        """保存检查点"""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
        }
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved at {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        if not os.path.exists(checkpoint_path):
            self.logger.warning(f"Checkpoint {checkpoint_path} does not exist")
            return

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.logger.info(f"Checkpoint loaded from {checkpoint_path}")

    def train_one_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0

        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {self.epoch}")
        for batch_idx, batch in enumerate(progress_bar):
            # 准备数据
            source_images = batch["image"].to(self.device)
            target_texts = batch["text"]
            image_names = batch.get("image_name", None)
            source_params = batch.get("params", None)
            source_rendered = batch.get("rendered_image", None)
            if source_params is not None:
                source_params = source_params.to(self.device)
            if source_rendered is not None:
                source_rendered = source_rendered.to(self.device)

            # 训练步骤
            with autocast():
                loss_dict = self.training_step(
                    source_images, target_texts,
                    source_params=source_params,
                    source_rendered=source_rendered,
                    image_names=image_names,
                )
                loss = loss_dict['loss']

            # NaN/Inf 检测: 跳过坏 step，避免污染权重
            if torch.isnan(loss) or torch.isinf(loss):
                self.logger.warning(
                    f"Step {self.global_step}: loss is NaN/Inf ({loss.item():.4f}), "
                    f"skipping optimizer step"
                )
                self.optimizer.zero_grad()
                continue

            # 反向传播
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()

            # 梯度裁剪 (必须在 unscale 后进行)
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm,
                error_if_nonfinite=False,
            )

            # 梯度 NaN/Inf 检测
            grad_is_finite = not (torch.isnan(grad_norm) or torch.isinf(grad_norm))
            if not grad_is_finite:
                self.logger.warning(
                    f"Step {self.global_step}: gradient norm is {grad_norm.item():.4f}, "
                    f"scaler will skip optimizer step"
                )

            # step + update 必须成对调用（即使梯度 NaN 也要调 update 清除 unscale 标记）
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # 只在梯度有效时更新训练状态
            if not grad_is_finite:
                continue

            total_loss += loss.item()
            self.global_step += 1

            # 更新进度条 (显示 loss 和 grad norm)
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "grad": f"{grad_norm.item():.2f}",
            })

            # 记录日志 (注入 grad_norm 便于诊断)
            if self.global_step % self.config.log_interval == 0:
                loss_dict["grad_norm"] = grad_norm.item()
                self.log_training(loss_dict, batch_idx)

            # 保存检查点
            if self.global_step % self.config.save_interval == 0:
                checkpoint_path = os.path.join(
                    self.config.checkpoint_dir,
                    f'checkpoint_step_{self.global_step}.pth'
                )
                self.save_checkpoint(checkpoint_path)

            # 评估 (eval_interval 设得很大时可被此条件跳过, 主要靠 epoch 结束时的 eval)
            if self.val_dataloader and self.global_step % self.config.eval_interval == 0:
                self.evaluate()

        # 每个 epoch 结束后评估一次
        if self.val_dataloader:
            self.evaluate()

        self.epoch += 1
        return total_loss / len(self.train_dataloader)

    def training_step(self, source_images, target_texts, **kwargs):
        """训练步骤，由子类实现"""
        raise NotImplementedError

    def evaluate(self):
        """评估，由子类实现"""
        raise NotImplementedError

    def log_training(self, loss_dict, batch_idx):
        """记录训练日志"""
        log_str = f"Epoch: {self.epoch}, Step: {self.global_step}, Batch: {batch_idx}"
        for key, value in loss_dict.items():
            if hasattr(value, 'item'):
                log_str += f", {key}: {value.item():.4f}"
            else:
                log_str += f", {key}: {value:.4f}"
        self.logger.info(log_str)

    def train(self):
        """训练循环"""
        self.logger.info("Starting training")

        # 恢复训练
        if hasattr(self.config, 'resume_from_checkpoint') and self.config.resume_from_checkpoint:
            self.load_checkpoint(self.config.resume_from_checkpoint)

        for epoch in range(self.epoch, self.config.num_epochs):
            avg_loss = self.train_one_epoch()
            self.logger.info(f"Epoch {epoch} completed. Average loss: {avg_loss:.4f}")

            # 保存最终检查点
            if epoch == self.config.num_epochs - 1:
                checkpoint_path = os.path.join(
                    self.config.checkpoint_dir,
                    f'final_checkpoint_epoch_{epoch}.pth'
                )
                self.save_checkpoint(checkpoint_path)

        self.logger.info("Training completed")