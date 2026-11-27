import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
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
        self.use_cuda_amp = torch.device(self.device).type == "cuda"
        self.amp_dtype = torch.bfloat16 if getattr(config, "use_bf16", False) else torch.float16
        if self.use_cuda_amp and self.amp_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 training requires GPU BF16 support")
        self.scaler = GradScaler(enabled=self.use_cuda_amp and self.amp_dtype == torch.float16)

        # 设置日志
        self.setup_logging()

        # 训练状态
        self.global_step = 0
        self.epoch = 0

    def configure_optimizers(self):
        """配置优化器"""
        groups = {}
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            learning_rate = self._optimizer_learning_rate(name)
            # Coupled Adam L2 can erase pretrained attention matrices whose task
            # gradients are small. Decouple decay, and leave biases/norms alone.
            decay = self.config.weight_decay if parameter.ndim > 1 else 0.0
            group = groups.setdefault((learning_rate, decay), {
                "params": [], "lr": learning_rate, "weight_decay": decay,
            })
            group["params"].append(parameter)
        return AdamW(
            list(groups.values()),
            lr=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2),
            weight_decay=self.config.weight_decay
        )

    def _optimizer_learning_rate(self, parameter_name):
        return self.config.learning_rate

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
            'model_state_dict': self.model.checkpoint_state(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
        }
        if hasattr(self.config, "guidance_scale"):
            checkpoint["inference_guidance_scale"] = self.config.guidance_scale
            checkpoint["inference_use_bf16"] = self.config.use_bf16
            checkpoint["inference_config"] = {
                name: getattr(self.config, name)
                for name in ("guidance_scale", "inference_strength", "num_inference_steps", "use_bf16",
                             "faceid_adapter_path", "faceid_encoder_path", "faceid_scale",
                             "use_photometric_head", "isolate_faceid_attention", "identity_gain_only",
                             "identity_refinement_rank")
                if hasattr(self.config, name)
            }
        from dataclasses import asdict, is_dataclass
        if is_dataclass(self.config):
            checkpoint["training_config"] = asdict(self.config)
        if hasattr(self, "_best_sampled_identity_score"):
            checkpoint["identity_validation"] = {
                "best_score": self._best_sampled_identity_score,
                "latest": getattr(self, "_last_sampled_identity_metrics", {}),
            }
            if hasattr(self, "_identity_calibration_expression_baseline"):
                checkpoint["identity_validation"]["calibration"] = {
                    "expression_baseline": self._identity_calibration_expression_baseline,
                    "max_expression_error": self._identity_calibration_max_expression_error,
                    "checks_without_improvement": getattr(
                        self, "_identity_calibration_checks_without_improvement", 0),
                    "last_validation_epoch": getattr(
                        self, "_identity_calibration_last_validation_epoch", None),
                    "patience": getattr(self, "_identity_calibration_patience", 5),
                }
                for key in ("baseline_progress", "min_expression_progress", "min_arcface_score"):
                    if hasattr(self, "_identity_calibration_" + key):
                        checkpoint["identity_validation"]["calibration"][key] = getattr(
                            self, "_identity_calibration_" + key)
        temporary_path = checkpoint_path + ".tmp"
        torch.save(checkpoint, temporary_path)
        os.replace(temporary_path, checkpoint_path)
        self.logger.info(f"Checkpoint saved at {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint {checkpoint_path} does not exist")

        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        saved_backend = checkpoint.get("training_config", {}).get("identity_loss_backend", "facenet")
        if saved_backend != getattr(self.config, "identity_loss_backend", "facenet"):
            raise ValueError("Cannot resume with a different identity loss backend; initialize weights for a new objective")
        if (saved_backend == "arcface" and checkpoint.get("training_config", {}).get(
                "identity_training_alignment", "generated") != getattr(
                    self.config, "identity_training_alignment", "generated")):
            raise ValueError("Cannot resume with a different identity training alignment; initialize a new objective")
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.model.load_checkpoint_state(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.restore_identity_validation(checkpoint)
        self.logger.info(f"Checkpoint loaded from {checkpoint_path}")

    def restore_identity_validation(self, checkpoint):
        if "identity_validation" in checkpoint:
            from math import isfinite
            score = checkpoint["identity_validation"].get("best_score")
            if score is not None and isfinite(float(score)):
                self._best_sampled_identity_score = score
            self._last_sampled_identity_metrics = checkpoint["identity_validation"].get("latest", {})
            calibration = checkpoint["identity_validation"].get("calibration", {})
            for key in ("expression_baseline", "max_expression_error", "checks_without_improvement",
                        "last_validation_epoch", "patience", "baseline_progress", "min_expression_progress",
                        "min_arcface_score"):
                if key in calibration:
                    setattr(self, "_identity_calibration_" + key, calibration[key])
            if calibration:
                self.stop_training = (getattr(self, "_identity_calibration_checks_without_improvement", 0)
                                      >= getattr(self, "_identity_calibration_patience", 5))

    def train_one_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        updated_steps = 0

        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {self.epoch}")
        for batch_idx, batch in enumerate(progress_bar):
            # 准备数据
            source_images = batch["image"].to(self.device)
            target_texts = batch["text"]
            image_names = batch.get("image_name", None)
            source_params = batch.get("params", None)
            source_rendered = batch.get("rendered_image", None)
            target_images = batch.get("target_image", None)
            target_params = batch.get("target_params", None)
            source_alignment = batch.get("source_alignment")
            if source_alignment is not None:
                source_alignment = source_alignment.to(self.device)
            target_alignment = batch.get("target_alignment")
            if target_alignment is not None:
                target_alignment = target_alignment.to(self.device)
            if source_params is not None:
                source_params = source_params.to(self.device)
            if source_rendered is not None:
                source_rendered = source_rendered.to(self.device)
            if target_images is not None:
                target_images = target_images.to(self.device)
            if target_params is not None:
                target_params = target_params.to(self.device)

            # 训练步骤
            with autocast(enabled=self.use_cuda_amp, dtype=self.amp_dtype):
                loss_dict = self.training_step(
                    source_images, target_texts,
                    source_params=source_params,
                    source_rendered=source_rendered,
                    image_names=image_names,
                    target_images=target_images,
                    target_params=target_params,
                    target_alignment=target_alignment,
                    source_alignment=source_alignment,
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
                    "skipping optimizer step"
                )

            # Clear the scaler's unscale state even when the optimizer update is skipped.
            if grad_is_finite:
                self.scaler.step(self.optimizer)
            self.scaler.update()

            # 只在梯度有效时更新训练状态
            if not grad_is_finite:
                continue

            total_loss += loss.item()
            self.global_step += 1
            updated_steps += 1

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
            if self.config.save_interval > 0 and self.global_step % self.config.save_interval == 0:
                checkpoint_path = os.path.join(
                    self.config.checkpoint_dir,
                    'latest_checkpoint.pth'
                )
                self.save_checkpoint(checkpoint_path)

            # 评估 (eval_interval 设得很大时可被此条件跳过, 主要靠 epoch 结束时的 eval)
            if (self.val_dataloader and self.config.eval_interval > 0
                    and self.global_step % self.config.eval_interval == 0):
                self.evaluate()

        if not updated_steps:
            raise RuntimeError("No optimizer steps completed during this epoch")

        # 每个 epoch 结束后评估一次
        if self.val_dataloader:
            self.evaluate()

        self.epoch += 1
        return total_loss / updated_steps

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
                # 跳过非标量张量 (如 generated_images, theta_target 等)
                try:
                    log_str += f", {key}: {value.item():.4f}"
                except RuntimeError:
                    continue
            elif isinstance(value, (int, float)):
                log_str += f", {key}: {value:.4f}"
        self.logger.info(log_str)

    def train(self):
        """训练循环"""
        self.logger.info("Starting training")

        # 恢复训练
        if hasattr(self.config, 'resume_from_checkpoint') and self.config.resume_from_checkpoint:
            self.load_checkpoint(self.config.resume_from_checkpoint)

        self.on_training_start()

        if getattr(self, "stop_training", False):
            self.logger.info("Restored calibration already reached its no-improvement limit")
            return

        for epoch in range(self.epoch, self.config.num_epochs):
            avg_loss = self.train_one_epoch()
            self.logger.info(f"Epoch {epoch} completed. Average loss: {avg_loss:.4f}")

            if getattr(self, "stop_training", False):
                self.save_checkpoint(os.path.join(self.config.checkpoint_dir, "stopped_no_improvement.pth"))
                self.logger.info("Stopping after sampled identity failed to improve; best checkpoint retained")
                break

            # 保存最终检查点
            if epoch == self.config.num_epochs - 1:
                checkpoint_path = os.path.join(
                    self.config.checkpoint_dir,
                    f'final_checkpoint_epoch_{epoch}.pth'
                )
                self.save_checkpoint(checkpoint_path)

        self.logger.info("Training completed")

    def on_training_start(self):
        """Run subclass initialization after every supported resume path."""
        pass
