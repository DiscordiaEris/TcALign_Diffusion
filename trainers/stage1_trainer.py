from .base_trainer import BaseTrainer
import torch
import torch.nn.functional as F


class Stage1Trainer(BaseTrainer):
    """第一阶段训练器：训练文本条件3D编辑器"""

    def __init__(self, config, model, train_dataloader, val_dataloader=None):
        super().__init__(config, model, train_dataloader, val_dataloader)
        self.num_epochs = config.num_epochs
        self._clip_cache = {}

    def _get_cached_clip_embedding(self, rendered, names):
        missing = [index for index, name in enumerate(names) if name not in self._clip_cache]
        if missing:
            embeddings = self.model.clip_encoder.encode_image(rendered[missing])
            self._clip_cache.update(
                (names[index], embedding.detach().cpu())
                for index, embedding in zip(missing, embeddings)
            )
        return torch.stack([self._clip_cache[name] for name in names]).to(rendered.device)

    def training_step(self, source_images, target_texts, source_params=None,
                      source_rendered=None, image_names=None, target_params=None, **kwargs):
        if target_params is None:
            raise ValueError("Stage 1 requires measured 3D coefficients from paired target portraits")
        editor = self.model.text_3d_editor
        with torch.no_grad():
            if source_params is None:
                source_params = self.model.get_3d_params(source_images)
            if source_rendered is None:
                source_rendered = self.model.render_from_params(source_params)
            if image_names is None:
                image_embedding = self.model.clip_encoder.encode_image(source_rendered)
            else:
                image_embedding = self._get_cached_clip_embedding(source_rendered, image_names)
            text_embedding = self.model.clip_encoder.encode_text(target_texts)
            attributes = [editor.parse_attributes(text) for text in target_texts]
            mask = editor.attribute_mask(attributes, source_params)
            reference = (target_params - source_params).masked_fill(~mask, 0)
            reference_embedding = self.model.clip_encoder.encode_image(
                self.model.render_from_params(source_params + reference)
            )

        delta = editor(image_embedding, text_embedding, target_texts)
        per_coefficient = F.smooth_l1_loss(delta, reference, reduction="none")
        # Weight each requested attribute equally, despite different block widths.
        block_losses = torch.stack([
            per_coefficient[:, group].mean(dim=-1) for group in editor.attribute_slices.values()
        ], dim=-1)
        loss_delta = (block_losses.sum(dim=-1) /
                      delta.new_tensor([len(group) for group in attributes])).mean()
        edited_embedding = self.model.clip_encoder.encode_image(
            self.model.render_from_params(source_params + delta)
        )
        loss_clip = 1 - F.cosine_similarity(edited_embedding.float(), reference_embedding.float()).mean()
        loss_l2 = delta.square().mean()
        return {
            "loss": loss_delta + 0.1 * loss_clip + self.config.delta_theta_l2_weight * loss_l2,
            "loss_delta": loss_delta,
            "loss_clip": loss_clip,
            "loss_l2": loss_l2,
        }

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        total = 0
        count = 0
        try:
            for batch in self.val_dataloader:
                source = batch["image"].to(self.device)
                losses = self.training_step(
                    source, batch["text"],
                    source_params=batch["params"].to(self.device),
                    source_rendered=batch["rendered_image"].to(self.device),
                    target_params=batch["target_params"].to(self.device),
                )
                total += losses["loss"].item() * len(source)
                count += len(source)
            value = total / count
            self.logger.info("Validation loss: %.4f", value)
            return value
        finally:
            self.model.train()
