import torch
import torch.nn as nn
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, DDIMScheduler
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKLOutput


class DiffusionUNetWrapper(nn.Module):
    """完整的扩散UNet包装器 - 包含VAE编码器/解码器和多模态输入支持"""

    def __init__(self, config):
        super().__init__()
        self.config = config

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
        self.unet = UNet2DConditionModel.from_pretrained(
            load_from,
            subfolder="unet",
            local_files_only=local_only
        )

        # 修改UNet输入通道数以接受多模态输入
        self._modify_unet_input_channels()

        # 冻结VAE参数（可选，根据需求决定是否冻结）
        if config.freeze_vae:
            for param in self.vae.parameters():
                param.requires_grad = False
            print("VAE parameters frozen")

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
            prediction_type=config.prediction_type if hasattr(config, 'prediction_type') else 'epsilon'
        )

        print(f"VAE latent channels: {self.vae.config.latent_channels}")
        print(f"UNet input channels: {self.unet.config.in_channels}")

    def _modify_unet_input_channels(self):
        """修改UNet输入通道数以接受多模态输入"""
        original_conv = self.unet.conv_in
        original_in_channels = original_conv.in_channels
        original_weight = original_conv.weight.clone()

        # 计算新的输入通道数
        # 原始输入通道数 + VAE潜在通道数 + RGB图像通道数
        new_in_channels = (
                original_in_channels +
                self.vae.config.latent_channels +  # 源图像潜在表示
                3  # 源图像RGB
        )

        print(f"Modifying UNet input channels: {original_in_channels} -> {new_in_channels}")

        # 创建新的卷积层
        new_conv = nn.Conv2d(
            new_in_channels,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None
        )

        # 初始化新权重
        nn.init.kaiming_uniform_(new_conv.weight, nonlinearity='relu')
        if new_conv.bias is not None:
            nn.init.zeros_(new_conv.bias)

        # 将原始权重复制到新卷积层的前面部分
        with torch.no_grad():
            new_conv.weight[:, :original_in_channels, :, :] = original_weight

        # 替换UNet的输入卷积层
        self.unet.conv_in = new_conv

        # 更新UNet的输入通道数配置
        self.unet.config.in_channels = new_in_channels

        print("UNet input channels modified successfully")

    def encode_image(self, images, return_dict=False):
        """
        编码图像到潜在空间

        参数:
            images: 输入图像 [batch_size, 3, H, W]，值范围 [-1, 1]
            return_dict: 是否返回完整的VAE输出字典

        返回:
            latents: 潜在表示 [batch_size, 4, H/8, W/8]
        """
        # 确保输入在正确范围内
        images = torch.clamp(images, -1, 1)

        # 使用VAE编码器
        with torch.no_grad():
            vae_output = self.vae.encode(images)

            if isinstance(vae_output, AutoencoderKLOutput):
                latents = vae_output.latent_dist.sample()
            else:
                # 兼容旧版本
                latents = vae_output[0] if isinstance(vae_output, tuple) else vae_output

            # 应用缩放因子
            latents = latents * self.vae.config.scaling_factor

        if return_dict and isinstance(vae_output, AutoencoderKLOutput):
            return latents, vae_output
        else:
            return latents

    def decode_latent(self, latents, return_dict=False):
        """
        从潜在空间解码图像

        参数:
            latents: 潜在表示 [batch_size, 4, H/8, W/8]
            return_dict: 是否返回完整的VAE输出字典

        返回:
            images: 重建图像 [batch_size, 3, H, W]，值范围 [-1, 1]
        """
        # 应用逆缩放因子
        latents = latents / self.vae.config.scaling_factor

        # 使用VAE解码器
        with torch.no_grad():
            vae_output = self.vae.decode(latents)

            if isinstance(vae_output, AutoencoderKLOutput):
                images = vae_output.sample
            else:
                # 兼容旧版本
                images = vae_output[0] if isinstance(vae_output, tuple) else vae_output

        # 确保输出在正确范围内
        images = torch.clamp(images, -1, 1)

        if return_dict and isinstance(vae_output, AutoencoderKLOutput):
            return images, vae_output
        else:
            return images

    def prepare_multimodal_input(self, noisy_latents, source_images, source_latents):
        """
        准备多模态输入

        参数:
            noisy_latents: 噪声潜在表示 [batch_size, 4, H_latent, W_latent]
            source_images: 源图像 [batch_size, 3, H, W]
            source_latents: 源图像的潜在表示 [batch_size, 4, H_latent, W_latent]

        返回:
            multimodal_input: 多模态输入 [batch_size, 11, H_latent, W_latent]
        """
        batch_size = noisy_latents.shape[0]

        # 调整源图像尺寸以匹配潜在表示尺寸
        H_latent, W_latent = noisy_latents.shape[2], noisy_latents.shape[3]
        source_images_resized = torch.nn.functional.interpolate(
            source_images,
            size=(H_latent, W_latent),
            mode='bilinear',
            align_corners=False
        )

        # 拼接多模态输入
        # 通道顺序: [噪声潜在(4) + 源潜在(4) + 源图像(3)] = 11通道
        multimodal_input = torch.cat([
            noisy_latents,
            source_latents,
            source_images_resized
        ], dim=1)

        return multimodal_input

    def forward(self, noisy_latents, timesteps, condition_embeddings, source_images, source_latents):
        """
        前向传播 - 接受多模态输入

        参数:
            noisy_latents: 噪声潜在表示 [batch_size, 4, H_latent, W_latent]
            timesteps: 时间步 [batch_size]
            condition_embeddings: 条件嵌入 [batch_size, seq_len, embed_dim]
            source_images: 源图像 [batch_size, 3, H, W]
            source_latents: 源图像的潜在表示 [batch_size, 4, H_latent, W_latent]

        返回:
            noise_pred: 预测的噪声 [batch_size, 4, H_latent, W_latent]
        """
        # 准备多模态输入
        multimodal_input = self.prepare_multimodal_input(
            noisy_latents, source_images, source_latents
        )

        # 预测噪声
        noise_pred = self.unet(
            multimodal_input,
            timesteps,
            encoder_hidden_states=condition_embeddings
        ).sample

        return noise_pred

    def add_noise(self, latents, noise, timesteps):
        """添加噪声到潜在表示"""
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        return noisy_latents

    def sample(
            self,
            condition_embeddings,
            source_images,
            source_latents,
            guidance_scale=7.5,
            num_inference_steps=50,
            generator=None
    ):
        """
        采样生成图像 - 接受多模态输入

        参数:
            condition_embeddings: 条件嵌入 [batch_size, seq_len, embed_dim]
            source_images: 源图像 [batch_size, 3, H, W]
            source_latents: 源图像的潜在表示 [batch_size, 4, H_latent, W_latent]
            guidance_scale: CFG引导尺度
            num_inference_steps: 推理步数
            generator: 随机数生成器

        返回:
            generated_latents: 生成的潜在表示 [batch_size, 4, H_latent, W_latent]
        """
        # 设置推理步数
        self.inference_scheduler.set_timesteps(num_inference_steps, device=condition_embeddings.device)

        # 初始化随机噪声 (与源图像潜在表示相同的形状)
        batch_size = condition_embeddings.shape[0]
        latents_shape = source_latents.shape
        latents = torch.randn(
            latents_shape,
            device=condition_embeddings.device,
            generator=generator
        )

        # CFG需要无条件嵌入
        uncond_embeddings = torch.zeros_like(condition_embeddings)

        # 迭代去噪
        for i, t in enumerate(self.inference_scheduler.timesteps):
            # 准备多模态输入
            multimodal_input = self.prepare_multimodal_input(latents, source_images, source_latents)

            # 扩展潜在代码以进行CFG
            latent_model_input = torch.cat([multimodal_input] * 2)
            timestep_model_input = torch.cat([t.expand(batch_size)] * 2)
            condition_model_input = torch.cat([uncond_embeddings, condition_embeddings])

            # 预测噪声
            noise_pred = self.unet(
                latent_model_input,
                timestep_model_input,
                encoder_hidden_states=condition_model_input
            ).sample

            # 执行CFG
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            # 计算上一个采样点 x_t -> x_t-1
            latents = self.inference_scheduler.step(
                noise_pred, t, latents, generator=generator
            ).prev_sample

            # 可选：打印进度
            if i % 10 == 0 or i == len(self.inference_scheduler.timesteps) - 1:
                print(f"Sampling step {i + 1}/{len(self.inference_scheduler.timesteps)}")

        return latents

    def compute_vae_loss(self, images, recon_images, kl_weight=1e-6):
        """
        计算VAE损失 (如果训练VAE)

        参数:
            images: 原始图像 [batch_size, 3, H, W]
            recon_images: 重建图像 [batch_size, 3, H, W]
            kl_weight: KL散度权重

        返回:
            vae_loss: VAE总损失
            recon_loss: 重建损失
            kl_loss: KL散度损失
        """
        # 编码图像
        vae_output = self.vae.encode(images)

        if isinstance(vae_output, AutoencoderKLOutput):
            latents = vae_output.latent_dist.sample()
            posterior = vae_output.latent_dist
        else:
            # 兼容旧版本
            latents = vae_output[0] if isinstance(vae_output, tuple) else vae_output
            posterior = None

        # 重建损失 (MSE)
        recon_loss = torch.nn.functional.mse_loss(recon_images, images)

        # KL散度损失
        kl_loss = 0.0
        if posterior is not None:
            kl_loss = posterior.kl().mean()

        # 总VAE损失
        vae_loss = recon_loss + kl_weight * kl_loss

        return vae_loss, recon_loss, kl_loss