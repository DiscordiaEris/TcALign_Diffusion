from pathlib import Path

import torch
import torch.nn as nn
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, DDIMScheduler
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKLOutput
from diffusers.models.autoencoders.vae import DecoderOutput
from transformers import CLIPTextModel, CLIPTokenizer
from .geometry_adapter import GeometryAdapter


class DiffusionUNetWrapper(nn.Module):
    """完整的扩散UNet包装器 - 包含VAE编码器/解码器，可选多模态输入"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.use_multimodal = getattr(config, 'use_multimodal_input', False)

        # 优先使用本地路径，否则从 HuggingFace Hub 加载
        load_from = config.diffusion_local_path if config.diffusion_local_path else config.diffusion_model_name
        local_only = bool(config.diffusion_local_path)

        # 加载VAE编码器/解码器
        print("Loading VAE...")
        self.vae = AutoencoderKL.from_pretrained(
            load_from,
            subfolder="vae",
            local_files_only=local_only
        )

        # 加载预训练的UNet
        print("Loading UNet...")
        unet_options = {}
        if local_only and (Path(load_from) / "unet/diffusion_pytorch_model.fp16.safetensors").is_file():
            unet_options["variant"] = "fp16"
        self.unet = UNet2DConditionModel.from_pretrained(
            load_from,
            subfolder="unet",
            local_files_only=local_only,
            **unet_options
        )
        # Use exactly the tokenizer/text encoder used to pretrain this UNet.
        # A generic CLIP sentence embedding is not a replacement for its 77
        # cross-attention tokens.
        self.tokenizer = CLIPTokenizer.from_pretrained(
            load_from, subfolder="tokenizer", local_files_only=local_only
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            load_from, subfolder="text_encoder", local_files_only=local_only
        )
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        # 多模态输入时扩展 UNet 输入通道（4→11）；否则保持标准 4 通道
        if self.use_multimodal:
            self._modify_unet_input_channels()
        else:
            print("Multimodal input disabled — UNet keeps standard 4 input channels")

        # 冻结VAE参数（可选，根据需求决定是否冻结）
        if config.freeze_vae:
            for param in self.vae.parameters():
                param.requires_grad = False
            print("VAE parameters frozen")

        # 选择性冻结 UNet：仅训练 cross-attention 层 + conv_in
        # UNet ~860M 参数全量训练会导致 Adam 状态 OOM (24 GB 显存不够)
        self._selectively_freeze_unet(config)
        self.unet.enable_gradient_checkpointing()
        self.geometry_adapter = None
        if getattr(config, "use_geometry_adapter", False):
            # SD 1.5's first three blocks accept residuals before downsampling.
            channels = []
            for block, width in zip(self.unet.down_blocks, self.unet.config.block_out_channels):
                if not getattr(block, "has_cross_attention", False):
                    break
                channels.append(width)
            if not channels:
                raise ValueError("The geometry adapter needs cross-attention down blocks")
            self.geometry_adapter = GeometryAdapter(channels)

        # 噪声调度器
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=config.num_train_timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            beta_schedule=config.beta_schedule,
            prediction_type=config.prediction_type if hasattr(config, 'prediction_type') else 'epsilon'
        )

        # 推理采样器
        self.inference_scheduler = DDIMScheduler(
            num_train_timesteps=config.num_train_timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            beta_schedule=config.beta_schedule,
            clip_sample=False,
            steps_offset=1,
            set_alpha_to_one=False,
            prediction_type=config.prediction_type if hasattr(config, 'prediction_type') else 'epsilon'
        )

        print(f"VAE latent channels: {self.vae.config.latent_channels}")
        print(f"UNet input channels: {self.unet.config.in_channels}")

    def _modify_unet_input_channels(self):
        """修改UNet输入通道数以接受多模态输入"""
        original_conv = self.unet.conv_in
        original_in_channels = original_conv.in_channels
        original_weight = original_conv.weight.clone()

        new_in_channels = (
                original_in_channels +
                self.vae.config.latent_channels +  # 源图像潜在表示
                3  # 源图像RGB
        )

        print(f"Modifying UNet input channels: {original_in_channels} -> {new_in_channels}")

        new_conv = nn.Conv2d(
            new_in_channels,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None
        )

        nn.init.kaiming_uniform_(new_conv.weight, nonlinearity='relu')
        if new_conv.bias is not None:
            nn.init.zeros_(new_conv.bias)

        with torch.no_grad():
            new_conv.weight[:, :original_in_channels, :, :] = original_weight
            # Preserve the original model at initialization. The newly added
            # source-image channels learn their contribution rather than
            # injecting random features into every denoising layer.
            new_conv.weight[:, original_in_channels:, :, :].zero_()
            if original_conv.bias is not None:
                new_conv.bias.copy_(original_conv.bias)

        self.unet.conv_in = new_conv
        self.unet.config.in_channels = new_in_channels

        print("UNet input channels modified successfully")

    def _selectively_freeze_unet(self, config):
        """选择性冻结 UNet 参数以节省显存。

        仅训练以下模块:
        1. conv_in — 多模态输入通道扩展层（随机初始化，必须训练）
        2. attn2 (cross-attention) — 接收 condition_embeddings 的交叉注意力层

        冻结所有其他参数（~95%），将可训练参数量从 ~860M 降到 ~120M，
        Adam 优化器状态从 ~12 GB 降到 ~2 GB。
        """
        # 先全部冻结
        for name, param in self.unet.named_parameters():
            param.requires_grad = False

        # 仅解冻 cross-attention (attn2) 和 conv_in
        trainable_names = []
        for name, param in self.unet.named_parameters():
            if 'attn2' in name or 'conv_in' in name:
                param.requires_grad = True
                trainable_names.append(name)

        trainable_count = sum(p.numel() for p in self.unet.parameters() if p.requires_grad)
        total_count = sum(p.numel() for p in self.unet.parameters())
        print(f"UNet selective freeze: {trainable_count:,}/{total_count:,} params trainable "
              f"({100*trainable_count/total_count:.1f}%), {len(trainable_names)} param groups unfrozen")

    def encode_image(self, images, return_dict=False, *, deterministic=False):
        """编码图像到潜在空间"""
        images = torch.clamp(images, -1, 1)

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
            images = images.float()
            vae_output = self.vae.encode(images)

            if isinstance(vae_output, AutoencoderKLOutput):
                latents = (vae_output.latent_dist.mode() if deterministic
                           else vae_output.latent_dist.sample())
            else:
                latents = vae_output[0] if isinstance(vae_output, tuple) else vae_output

            latents = latents * self.vae.config.scaling_factor

        if return_dict and isinstance(vae_output, AutoencoderKLOutput):
            return latents, vae_output
        else:
            return latents

    @torch.no_grad()
    def encode_prompt(self, texts):
        """Return native Stable Diffusion CLIP token embeddings [B, 77, 768]."""
        tokens = self.tokenizer(
            texts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        tokens = {name: value.to(next(self.unet.parameters()).device) for name, value in tokens.items()}
        return self.text_encoder(input_ids=tokens["input_ids"]).last_hidden_state

    def decode_latent(self, latents, return_dict=False):
        """从潜在空间解码图像（无梯度，用于推理/监控）"""
        latents = latents / self.vae.config.scaling_factor

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
            latents = latents.float()
            vae_output = self.vae.decode(latents)

            if isinstance(vae_output, (AutoencoderKLOutput, DecoderOutput)):
                images = vae_output.sample
            else:
                images = vae_output[0] if isinstance(vae_output, tuple) else vae_output

        images = torch.clamp(images, -1, 1)

        if return_dict and isinstance(vae_output, (AutoencoderKLOutput, DecoderOutput)):
            return images, vae_output
        else:
            return images

    def decode_latent_train(self, latents):
        """从潜在空间解码图像（保留梯度，VAE decoder 冻结但梯度可穿过）

        用于 loss_clip 等需要梯度回传 UNet 的编辑损失。
        始终在 fp32 下运行以避免 VAE decoder 的 fp16 dtype 不匹配。
        使用 gradient checkpointing 节省显存：forward 时不保留中间激活，backward 时重算。

        只返回图像 tensor，不返回 VAE output dict。
        """
        latents = latents / self.vae.config.scaling_factor

        def _vae_decode(z):
            with torch.cuda.amp.autocast(enabled=False):
                vae_output = self.vae.decode(z.float())
            if isinstance(vae_output, (AutoencoderKLOutput, DecoderOutput)):
                return vae_output.sample
            else:
                return vae_output[0] if isinstance(vae_output, tuple) else vae_output

        # gradient checkpointing: forward 不存中间激活，backward 时重算 VAE decoder
        # 显存节省 ~2-4 GiB (VAE decoder 中间特征图)，代价是 backward 额外一次 forward
        images = torch.utils.checkpoint.checkpoint(
            _vae_decode, latents, use_reentrant=False
        )

        images = torch.clamp(images, -1, 1)
        return images

    def _prepare_multimodal_input(self, noisy_latents, source_images, source_latents):
        """准备多模态输入: [噪声潜在(4) + 源潜在(4) + 源图像(3)] = 11通道"""
        H_latent, W_latent = noisy_latents.shape[2], noisy_latents.shape[3]
        source_images_resized = torch.nn.functional.interpolate(
            source_images,
            size=(H_latent, W_latent),
            mode='bilinear',
            align_corners=False
        )
        return torch.cat([noisy_latents, source_latents, source_images_resized], dim=1)

    def geometry_features(self, geometry, latent_size):
        if self.geometry_adapter is None or geometry is None:
            return None
        return self.geometry_adapter(geometry, latent_size)

    def forward(self, noisy_latents, timesteps, condition_embeddings, source_images, source_latents,
                *, geometry_residuals=None):
        """
        前向传播

        参数:
            noisy_latents: 噪声潜在表示 [batch_size, 4, H_latent, W_latent]
            timesteps: 时间步 [batch_size]
            condition_embeddings: 条件嵌入 [batch_size, seq_len, embed_dim]
            source_images: 源图像 [batch_size, 3, H, W]
            source_latents: 源图像的潜在表示 [batch_size, 4, H_latent, W_latent]

        返回:
            noise_pred: 预测的噪声 [batch_size, 4, H_latent, W_latent]
        """
        if self.use_multimodal:
            unet_input = self._prepare_multimodal_input(noisy_latents, source_images, source_latents)
        else:
            unet_input = noisy_latents

        noise_pred = self.unet(
            unet_input,
            timesteps,
            encoder_hidden_states=condition_embeddings,
            # Diffusers consumes the list with pop(), so pass a fresh list.
            down_intrablock_additional_residuals=(
                list(geometry_residuals) if geometry_residuals is not None else None
            ),
        ).sample

        return noise_pred

    def add_noise(self, latents, noise, timesteps):
        """添加噪声到潜在表示"""
        return self.noise_scheduler.add_noise(latents, noise, timesteps)

    @torch.no_grad()
    def sample(self, condition_embeddings, source_images, source_latents,
               guidance_scale=7.5, num_inference_steps=50, generator=None, strength=1.0,
               geometry=None):
        if not 0 < strength <= 1:
            raise ValueError("Image-to-image strength must be in (0, 1]")
        if num_inference_steps < 1:
            raise ValueError("At least one DDIM step is required")
        scheduler = self.inference_scheduler
        scheduler.set_timesteps(num_inference_steps, device=source_latents.device)
        timesteps = scheduler.timesteps
        batch_size = len(source_latents)
        noise = torch.randn(source_latents.shape, device=source_latents.device,
                            dtype=source_latents.dtype, generator=generator)
        if strength < 1:
            active_steps = max(1, int(strength * len(timesteps)))
            timesteps = timesteps[-active_steps:]
            latents = scheduler.add_noise(source_latents, noise, timesteps[0].expand(batch_size))
        else:
            latents = noise * scheduler.init_noise_sigma

        unconditional = self.encode_prompt([""] * batch_size) if guidance_scale != 1 else None
        use_bf16 = latents.is_cuda and getattr(self.config, "use_bf16", False) and torch.cuda.is_bf16_supported()
        amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
        with torch.cuda.amp.autocast(enabled=latents.is_cuda, dtype=amp_dtype):
            geometry_residuals = self.geometry_features(geometry, latents.shape[-2:])
        for timestep in timesteps:
            t = timestep.expand(batch_size)
            with torch.cuda.amp.autocast(enabled=latents.is_cuda, dtype=amp_dtype):
                noise_pred = self(latents, t, condition_embeddings, source_images, source_latents,
                                  geometry_residuals=geometry_residuals)
                if unconditional is not None:
                    baseline = self(latents, t, unconditional, source_images, source_latents,
                                    geometry_residuals=geometry_residuals)
                    noise_pred = baseline + guidance_scale * (noise_pred - baseline)
            latents = scheduler.step(noise_pred.float(), timestep, latents, generator=generator).prev_sample
        return latents

    def compute_vae_loss(self, images, recon_images, kl_weight=1e-6):
        posterior = self.vae.encode(images).latent_dist
        recon_loss = torch.nn.functional.mse_loss(recon_images, images)
        kl_loss = posterior.kl().mean()
        return recon_loss + kl_weight * kl_loss, recon_loss, kl_loss
