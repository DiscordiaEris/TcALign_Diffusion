from .base_trainer import BaseTrainer
import torch
import torch.nn.functional as F


class Stage1Trainer(BaseTrainer):
    """第一阶段训练器：训练文本条件3D编辑器"""

    def __init__(self, config, model, train_dataloader, val_dataloader=None):
        super().__init__(config, model, train_dataloader, val_dataloader)
        self.num_epochs = config.num_epochs
        # CLIP 嵌入缓存: 渲染图用确定性变换，嵌入可复用
        self._clip_cache = {}  # image_name → CLIP embedding tensor [1, 768]

    def _get_cached_clip_embedding(self, rendered_0, image_names):
        """获取渲染图的 CLIP 嵌入，优先从缓存读取"""
        unique_names = set(image_names)
        uncached = [n for n in image_names if n not in self._clip_cache]
        if uncached:
            uncached_set = set(uncached)
            indices = [i for i, n in enumerate(image_names) if n in uncached_set]
            for i in indices:
                emb = self.model.clip_encoder.encode_image(rendered_0[i:i + 1])
                self._clip_cache[image_names[i]] = emb.detach()
        embeddings = [self._clip_cache[n] for n in image_names]
        return torch.cat(embeddings, dim=0)

    def training_step(self, source_images, target_texts,
                       source_params=None, source_rendered=None,
                       image_names=None):
        """训练步骤 - 复用预计算的3D参数和渲染图"""
        with torch.no_grad():
            if source_params is not None and source_rendered is not None:
                theta_0 = source_params
                rendered_0 = source_rendered
            else:
                theta_0 = self.model.get_3d_params(source_images)
                rendered_0 = self.model.render_from_params(theta_0)

            # 渲染图 CLIP 嵌入（原始论文设计：CLIP 对 3D 渲染人脸也有合理的编码能力）
            if image_names is not None:
                e_img = self._get_cached_clip_embedding(rendered_0, image_names)
            else:
                e_img = self.model.clip_encoder.encode_image(rendered_0)
            e_txt = self.model.clip_encoder.encode_text(target_texts)

        # 通过3D编辑器预测3D参数变化，同时获取AES操控后的嵌入
        delta_theta, e_manipulated = self.model.text_3d_editor(
            e_img, e_txt, target_texts, return_manipulated=True
        )

        # 目标3D参数
        theta_target = theta_0 + delta_theta
        theta_target = torch.clamp(
            theta_target,
            min=-self.config.delta_theta_clamp * 2,
            max=self.config.delta_theta_clamp * 2,
        )

        # 渲染目标图像
        rendered_target = self.model.render_from_params(theta_target)

        # Loss: 渲染结果 CLIP 嵌入 应接近 AES 操控嵌入 (保留源图身份, 只改属性)
        e_rendered_target = self.model.clip_encoder.encode_image(rendered_target)
        loss_clip = 1 - F.cosine_similarity(e_rendered_target, e_manipulated, dim=-1).mean()

        # 诊断: 源渲染图 vs 文本 (基线) / 目标渲染图 vs 操控嵌入
        with torch.no_grad():
            sim_baseline = F.cosine_similarity(e_img, e_txt, dim=-1).mean()
            sim_target = 1 - loss_clip

        # Δθ L2 正则化: 防止 delta_theta 无限膨胀导致渲染器 NaN
        loss_l2 = delta_theta.pow(2).mean()

        # 自适应 L2 权重: loss_l2 过高时进一步加大惩罚
        l2_weight = self.config.delta_theta_l2_weight
        if loss_l2 > 10.0:
            l2_weight *= 2.0  # loss_l2 超标时加倍惩罚

        loss = loss_clip + l2_weight * loss_l2

        return {
            "loss": loss,
            "loss_clip": loss_clip,
            "loss_l2": loss_l2,
            "l2_weight": l2_weight,
            "sim_base": sim_baseline,   # 源图-文本 cos_sim
            "sim_target": sim_target,   # 目标渲染-文本 cos_sim
        }

    def evaluate(self):
        """评估 - 复用预计算数据"""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_dataloader:
                source_images = batch["image"].to(self.device)
                target_texts = batch["text"]
                image_names = batch.get("image_name")
                source_params = batch.get("params")
                source_rendered = batch.get("rendered_image")
                if source_params is not None:
                    source_params = source_params.to(self.device)
                if source_rendered is not None:
                    source_rendered = source_rendered.to(self.device)

                if source_params is not None:
                    theta_0 = source_params
                    rendered_0 = source_rendered
                else:
                    theta_0 = self.model.get_3d_params(source_images)
                    rendered_0 = self.model.render_from_params(theta_0)

                e_img = self.model.clip_encoder.encode_image(rendered_0)
                e_txt = self.model.clip_encoder.encode_text(target_texts)

                delta_theta, e_manipulated = self.model.text_3d_editor(
                    e_img, e_txt, target_texts, return_manipulated=True
                )

                theta_target = theta_0 + delta_theta
                rendered_target = self.model.render_from_params(theta_target)
                e_rendered_target = self.model.clip_encoder.encode_image(rendered_target)
                loss = 1 - F.cosine_similarity(e_rendered_target, e_manipulated, dim=-1).mean()
                total_loss += loss.item()
                num_batches += 1

        if num_batches > 0:
            avg_loss = total_loss / num_batches
            self.logger.info(f"Validation loss: {avg_loss:.4f}")
        else:
            avg_loss = 0.0
            self.logger.info("No validation data available, skipping evaluation")

        self.model.train()
        return avg_loss