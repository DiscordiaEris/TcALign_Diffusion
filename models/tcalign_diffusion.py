import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .clip_encoder import CLIPEncoder
from .text_conditional_3d_editor import TextConditional3DEditor
from .conditional_diffusion_mapping_network import ConditionalDiffusionMappingNetwork
from .diffusion_unet_wrapper import DiffusionUNetWrapper
from utils.deep3d_utils import load_deep3d_model
from utils.pytorch3d_renderer import PyTorch3DRenderer


class TcAlignDiffusion(nn.Module):
    """TcALign with Diffusion 主模型 - 完整集成VAE和多模态输入"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = config.device

        # 预训练模型
        self.clip_encoder = CLIPEncoder(
            config.clip_model_name, config.device, local_path=getattr(config, 'clip_local_path', '')
        )

        # 3D处理器和渲染器
        self.deep3d_processor = load_deep3d_model(config)
        self.renderer = PyTorch3DRenderer(config)

        # 可训练模块
        self.text_3d_editor = TextConditional3DEditor(config)

        # 扩散模块延迟加载 (Stage 1 不需要, 避免无网络时下载 SD 模型)
        self.diffusion_mapper = None
        self.diffusion_unet = None

        print("TcAlignDiffusion model initialized successfully")
        print(f"Using multimodal input: {config.use_multimodal_input}")

    def _init_diffusion_modules(self):
        """延迟初始化扩散模块 (Stage 2 或推理时首次调用)"""
        if self.diffusion_mapper is None:
            self.diffusion_mapper = ConditionalDiffusionMappingNetwork(self.config)
        if self.diffusion_unet is None:
            self.diffusion_unet = DiffusionUNetWrapper(self.config)

    def get_3d_params(self, image_tensor):
        """从图像获取3D参数（支持batch）"""
        # 确保图像在正确的范围内 [-1, 1]
        image_tensor = torch.clamp(image_tensor, -1, 1)

        batch_params = []
        # Deep3DProcessor.process 目前对单张图像设计
        for i in range(image_tensor.shape[0]):
            params = self.deep3d_processor.process(image_tensor[i:i+1])
            all_params = np.concatenate([
                params['shape'].flatten(),
                params['expression'].flatten(),
                params['pose'].flatten(),
                params['texture'].flatten(),
                params['lighting'].flatten()
            ])
            batch_params.append(torch.FloatTensor(all_params))

        return torch.stack(batch_params).to(self.device)

    def render_from_params(self, params):
        """从3D参数渲染图像 (params: [batch, total_3d_dim])

        直接使用 PyTorch3D 可微渲染管线，保留梯度用于 end-to-end 训练。
        """
        sd = self.config.shape_dim
        ed = self.config.expression_dim
        pd = self.config.pose_dim
        td = self.config.texture_dim
        ld = self.config.lighting_dim

        # 直接构建 tensor 参数字典，不做 detach → 梯度可以通过渲染器回传
        tensor_params = {
            'shape': params[:, :sd],
            'expression': params[:, sd:sd + ed],
            'pose': params[:, sd + ed:sd + ed + pd],
            'texture': params[:, sd + ed + pd:sd + ed + pd + td],
            'lighting': params[:, sd + ed + pd + td:sd + ed + pd + td + ld],
        }

        return self.renderer.render(tensor_params)

    def encode_source_images(self, source_images):
        """编码源图像到潜在空间"""
        self._init_diffusion_modules()
        source_latents = self.diffusion_unet.encode_image(source_images)
        return source_latents

    def forward(self, source_images, target_texts, training_mode="diffusion", theta_0=None):
        """
        前向传播

        参数:
            source_images: 源图像 [batch_size, C, H, W]
            target_texts: 目标文本列表 [batch_size]
            training_mode: "3d_editor" 或 "diffusion"
            theta_0: 预计算的 3D 参数 [batch, total_3d_dim]，跳过 Deep3D 推理
        """
        if training_mode == "3d_editor":
            return self.forward_3d_editor(source_images, target_texts)
        elif training_mode == "diffusion":
            return self.forward_diffusion(source_images, target_texts, theta_0=theta_0)
        else:
            raise ValueError(f"Unknown training mode: {training_mode}")

    def forward_3d_editor(self, source_images, target_texts):
        """3D编辑器前向传播"""
        # 获取源图像的3D参数
        theta_0 = self.get_3d_params(source_images)

        # 获取源图像的渲染图
        rendered_0 = self.render_from_params(theta_0)

        # 提取CLIP嵌入
        with torch.no_grad():
            e_img = self.clip_encoder.encode_image(rendered_0)
            e_txt = self.clip_encoder.encode_text(target_texts)

        # 通过3D编辑器预测3D参数变化
        delta_theta = self.text_3d_editor(e_img, e_txt, target_texts)

        return {"delta_theta": delta_theta, "theta_0": theta_0}

    def forward_diffusion(self, source_images, target_texts, theta_0=None):
        """扩散模型前向传播 - 完整使用VAE和多模态输入

        参数:
            theta_0: 预计算的 3D 参数 [batch, total_3d_dim]。传入则跳过 Deep3D 推理。
        """
        self._init_diffusion_modules()
        batch_size = source_images.shape[0]

        # 获取源图像的3D参数 (优先使用预计算值)
        with torch.no_grad():
            if theta_0 is None:
                theta_0 = self.get_3d_params(source_images)

        # 获取源图像的渲染图
        rendered_0 = self.render_from_params(theta_0)

        # 提取CLIP嵌入
        with torch.no_grad():
            e_img = self.clip_encoder.encode_image(rendered_0)
            e_txt = self.clip_encoder.encode_text(target_texts)

        # 通过3D编辑器预测目标3D参数
        delta_theta_pred = self.text_3d_editor(e_img, e_txt, target_texts)
        theta_target = theta_0 + delta_theta_pred

        # 计算Δθ
        delta_theta = theta_target - theta_0

        # 编码源图像到潜在空间
        source_latents = self.encode_source_images(source_images)

        # 通过扩散映射网络获取条件嵌入
        condition_embeddings = self.diffusion_mapper(delta_theta, source_latents)

        # 准备扩散过程
        noise = torch.randn_like(source_latents)
        timesteps = torch.randint(
            0, self.config.num_train_timesteps,
            (batch_size,), device=self.device
        ).long()

        # 添加噪声
        noisy_latents = self.diffusion_unet.add_noise(source_latents, noise, timesteps)

        # 预测噪声 (使用多模态输入)
        noise_pred = self.diffusion_unet(
            noisy_latents, timesteps, condition_embeddings, source_images, source_latents
        )

        # 基础扩散损失
        loss_diffusion = F.mse_loss(noise_pred, noise)

        # 3D一致性损失
        with torch.no_grad():
            # 解码生成图像
            generated_images = self.diffusion_unet.decode_latent(noisy_latents)

            # 预测生成图像的3D参数
            theta_pred = self.get_3d_params(generated_images)

        loss_3d = F.mse_loss(theta_pred, theta_target)

        # 身份保持损失 (使用ArcFace或像素级相似度)
        loss_id = F.mse_loss(generated_images, source_images)

        # CLIP对齐损失
        with torch.no_grad():
            clip_emb_gen = self.clip_encoder.encode_image(generated_images)

        loss_clip = 1 - F.cosine_similarity(clip_emb_gen, e_txt, dim=-1).mean()

        # VAE损失 (如果训练VAE)
        loss_vae = torch.tensor(0.0, device=self.device)
        if not self.config.freeze_vae:
            loss_vae, _, _ = self.diffusion_unet.compute_vae_loss(
                source_images, generated_images, self.config.vae_kl_weight
            )

        # 总损失
        total_loss = (
                self.config.lambda_diffusion * loss_diffusion +
                self.config.lambda_3d * loss_3d +
                self.config.lambda_id * loss_id +
                self.config.lambda_clip * loss_clip +
                self.config.lambda_vae * loss_vae
        )

        return {
            "loss": total_loss,
            "loss_diffusion": loss_diffusion,
            "loss_3d": loss_3d,
            "loss_id": loss_id,
            "loss_clip": loss_clip,
            "loss_vae": loss_vae,
            "generated_images": generated_images,
            "theta_target": theta_target
        }

    @torch.no_grad()
    def generate(self, source_images, target_texts, guidance_scale=7.5, num_inference_steps=50):
        """生成图像 - 完整使用VAE和多模态输入"""
        self._init_diffusion_modules()
        batch_size = source_images.shape[0]

        # 获取源图像的3D参数
        theta_0 = self.get_3d_params(source_images)

        # 获取源图像的渲染图
        rendered_0 = self.render_from_params(theta_0)

        # 提取CLIP嵌入
        e_img = self.clip_encoder.encode_image(rendered_0)
        e_txt = self.clip_encoder.encode_text(target_texts)

        # 通过3D编辑器预测目标3D参数
        delta_theta_pred = self.text_3d_editor(e_img, e_txt, target_texts)
        theta_target = theta_0 + delta_theta_pred

        # 计算Δθ
        delta_theta = theta_target - theta_0

        # 编码源图像到潜在空间
        source_latents = self.encode_source_images(source_images)

        # 通过扩散映射网络获取条件嵌入
        condition_embeddings = self.diffusion_mapper(delta_theta, source_latents)

        # 使用扩散模型采样生成图像 (使用多模态输入)
        generated_latents = self.diffusion_unet.sample(
            condition_embeddings,
            source_images,
            source_latents,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps
        )

        # 解码潜在代码
        generated_images = self.diffusion_unet.decode_latent(generated_latents)

        # 使用3D预测器生成最终参数
        final_theta = self.get_3d_params(generated_images)

        return generated_images, final_theta

    def train_vae_only(self, source_images):
        """仅训练VAE (如果VAE未冻结)"""
        self._init_diffusion_modules()
        if self.config.freeze_vae:
            print("VAE is frozen, skipping VAE-only training")
            return torch.tensor(0.0, device=self.device)

        # 编码-解码流程
        source_latents = self.encode_source_images(source_images)
        reconstructed_images = self.diffusion_unet.decode_latent(source_latents)

        # 计算VAE损失
        vae_loss, recon_loss, kl_loss = self.diffusion_unet.compute_vae_loss(
            source_images, reconstructed_images, self.config.vae_kl_weight
        )

        return {
            "loss": vae_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "reconstructed_images": reconstructed_images
        }