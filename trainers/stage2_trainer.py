from .base_trainer import BaseTrainer
import torch
import torch.nn.functional as F
import os
import numpy as np


class Stage2Trainer(BaseTrainer):
    """第二阶段训练器：训练条件扩散映射网络 - 支持多模态输入"""

    def __init__(self, config, model, train_dataloader, val_dataloader=None):
        super().__init__(config, model, train_dataloader, val_dataloader)
        self.num_epochs = config.num_epochs

        # 加载第一阶段训练好的模型
        if hasattr(config, 'stage1_checkpoint_path') and config.stage1_checkpoint_path:
            checkpoint = torch.load(config.stage1_checkpoint_path, map_location=self.device)
            full_state_dict = checkpoint['model_state_dict']
            # 只提取 text_3d_editor 子模块的权重 (stage1 checkpoint 保存了整个模型)
            text_3d_editor_state = {
                k.replace('text_3d_editor.', ''): v
                for k, v in full_state_dict.items()
                if k.startswith('text_3d_editor.')
            }
            if text_3d_editor_state:
                self.model.text_3d_editor.load_state_dict(text_3d_editor_state)
                self.logger.info(f"Loaded stage1 text_3d_editor from {config.stage1_checkpoint_path}")
            else:
                # 兼容直接保存 text_3d_editor state_dict 的情况
                self.model.text_3d_editor.load_state_dict(full_state_dict, strict=False)
                self.logger.info(f"Loaded stage1 checkpoint (strict=False) from {config.stage1_checkpoint_path}")

        # 冻结第一阶段模型
        for param in self.model.text_3d_editor.parameters():
            param.requires_grad = False

    def training_step(self, source_images, target_texts,
                       source_params=None, source_rendered=None, **kwargs):
        """训练步骤 - 使用预计算参数避免重复跑 Deep3D"""
        loss_dict = self.model(
            source_images, target_texts, training_mode="diffusion",
            theta_0=source_params,  # 传入预计算的 3D 参数
        )
        return loss_dict

    def evaluate(self):
        """评估 - 使用预计算参数避免 Deep3D"""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_dataloader:
                source_images = batch["image"].to(self.device)
                target_texts = batch["text"]
                source_params = batch.get("params")
                if source_params is not None:
                    source_params = source_params.to(self.device)

                # 计算损失 (传入预计算的 3D 参数跳过 Deep3D)
                loss_dict = self.model(
                    source_images, target_texts, training_mode="diffusion",
                    theta_0=source_params,
                )
                total_loss += loss_dict['loss'].item()
                num_batches += 1

        if num_batches > 0:
            avg_loss = total_loss / num_batches
            self.logger.info(f"Validation loss: {avg_loss:.4f}")
            # 生成示例图像
            self.generate_examples()
        else:
            avg_loss = 0.0
            self.logger.info("No validation data available, skipping evaluation")

        self.model.train()
        return avg_loss

    def generate_examples(self):
        """生成示例图像"""
        # 从验证集获取一批数据
        batch = next(iter(self.val_dataloader))
        source_images = batch["image"].to(self.device)[:4]  # 取前4个
        target_texts = batch["text"][:4]

        # 生成图像
        with torch.no_grad():
            generated_images, final_theta = self.model.generate(
                source_images, target_texts,
                guidance_scale=self.config.guidance_scale,
                num_inference_steps=self.config.num_inference_steps
            )

        # 保存图像
        from utils.visualization import save_image_grid
        output_path = os.path.join(
            self.config.output_dir,
            f"examples_step_{self.global_step}.png"
        )
        save_image_grid(source_images, generated_images, target_texts, output_path)

        # 保存3D参数
        theta_path = os.path.join(
            self.config.output_dir,
            f"theta_step_{self.global_step}.npy"
        )
        np.save(theta_path, final_theta.cpu().numpy())

        self.logger.info(f"Example images saved at {output_path}")
        self.logger.info(f"3D parameters saved at {theta_path}")