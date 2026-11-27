import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.deep3d_utils import load_deep3d_model
from utils.pytorch3d_renderer import PyTorch3DRenderer
from utils.losses import (make_edit_training_pair, requested_attribute_control_loss,
                         requested_lighting_exposure_loss)
from .clip_encoder import CLIPEncoder
from .conditional_diffusion_mapping_network import ConditionalDiffusionMappingNetwork
from .diffusion_unet_wrapper import DiffusionUNetWrapper
from .geometry_adapter import align_geometry
from .text_conditional_3d_editor import TextConditional3DEditor


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
        self.identity_loss = None
        self.photometric_head = None

        print("TcAlignDiffusion model initialized successfully")
        print(f"Using multimodal input: {config.use_multimodal_input}")

    def _init_diffusion_modules(self):
        """延迟初始化扩散模块 (Stage 2 或推理时首次调用)"""
        if self.diffusion_mapper is None:
            self.diffusion_mapper = ConditionalDiffusionMappingNetwork(self.config).to(self.device)
        if self.diffusion_unet is None:
            self.diffusion_unet = DiffusionUNetWrapper(self.config).to(self.device)
        if getattr(self.config, "use_photometric_head", False) and self.photometric_head is None:
            from .photometric_exposure import PhotometricExposureHead

            self.photometric_head = PhotometricExposureHead(self.config.condition_embed_dim).to(self.device)
        self.configure_identity_gain_training()

    def configure_identity_gain_training(self):
        if not getattr(self.config, "identity_gain_only", False):
            return
        if hasattr(self, "_identity_backbone_checkpoint_keys"):
            return
        # Frozen fine-tuned weights still belong to this checkpoint. Rebuilding
        # them from the original SD prior would change the calibrated model.
        self._identity_backbone_checkpoint_keys = {
            name for name, parameter in self.named_parameters() if parameter.requires_grad
        }
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(".identity_refinement." in name if getattr(
                self.config, "identity_refinement_rank", 0) else name.endswith(".identity_gain_logit"))

    def _checkpoint_keys(self):
        return ({name for name, param in self.named_parameters()
                 if param.requires_grad or name.startswith("text_3d_editor.")}
                | getattr(self, "_identity_backbone_checkpoint_keys", set()))

    def initialize_identity_loss(self):
        if self.config.lambda_id and self.identity_loss is None:
            if getattr(self.config, "identity_loss_backend", "facenet") == "arcface":
                from .arcface_identity_loss import ArcFaceIdentityLoss

                self.identity_loss = ArcFaceIdentityLoss(
                    self.config.identity_arcface_weights_path, self.deep3d_processor,
                    training_alignment=getattr(self.config, "identity_training_alignment", "generated"),
                ).to(self.device).eval()
            else:
                from .face_identity_loss import FaceIdentityLoss

                self.identity_loss = FaceIdentityLoss(
                    self.config.identity_weights_path,
                    normalize_exposure=self.config.identity_exposure_normalization,
                ).to(self.device).eval()

    def checkpoint_state(self):
        keys = self._checkpoint_keys()
        return {name: tensor for name, tensor in self.state_dict().items() if name in keys}

    def load_checkpoint_state(self, state):
        expected = self._checkpoint_keys()
        missing = expected - state.keys()
        unknown = state.keys() - self.state_dict().keys()
        if missing or unknown:
            raise RuntimeError(f"Incompatible checkpoint: missing {sorted(missing)}, "
                               f"unknown {sorted(unknown)}")
        self.load_state_dict(state, strict=False)

    def get_3d_params(self, image_tensor, *, return_alignment=False):
        """从图像获取3D参数（支持batch）"""
        # 确保图像在正确的范围内 [-1, 1]
        image_tensor = torch.clamp(image_tensor, -1, 1)

        batch_params = []
        alignments = []
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
            if return_alignment:
                alignments.append(torch.from_numpy(params["alignment"]))

        coefficients = torch.stack(batch_params).to(self.device)
        if return_alignment:
            return coefficients, torch.stack(alignments).to(self.device)
        return coefficients

    def render_from_params(self, params, *, geometry=False):
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

        return self.renderer.render(tensor_params, geometry=geometry)

    def encode_source_images(self, source_images):
        """编码源图像到潜在空间"""
        self._init_diffusion_modules()
        source_latents = self.diffusion_unet.encode_image(source_images, deterministic=True)
        return source_latents

    def encode_source_identity(self, source_images):
        if not getattr(self.config, "faceid_adapter_path", ""):
            return None
        if not hasattr(self, "faceid_encoder"):
            from .faceid_adapter import FaceIDSourceEncoder

            self.faceid_encoder = FaceIDSourceEncoder(self.config.faceid_encoder_path, self.deep3d_processor)
        return self.faceid_encoder.encode(source_images)

    def calibrate_exposure(self, source, generated, texts, tokens, alignment):
        if getattr(self, "photometric_head", None) is None:
            return generated
        from .photometric_exposure import apply_photometric_exposure, explicit_lighting_requests

        return apply_photometric_exposure(
            self.photometric_head, source, generated, tokens, alignment,
            explicit_lighting_requests(texts, device=generated.device),
        )

    def train(self, mode=True):
        super().train(mode)
        self.clip_encoder.eval()
        if not any(p.requires_grad for p in self.text_3d_editor.parameters()):
            self.text_3d_editor.eval()
        if getattr(self.config, "identity_gain_only", False):
            # Calibration changes gains, never frozen normalization buffers or
            # dropout behavior relative to the deployed generator.
            for module in self.children():
                module.eval()
        return self

    def predict_delta(self, theta_0, texts, rendered=None):
        if rendered is None:
            rendered = self.render_from_params(theta_0)
        with torch.no_grad():
            image_embedding = self.clip_encoder.encode_image(rendered)
            text_embedding = self.clip_encoder.encode_text(texts)
        return self.text_3d_editor(image_embedding, text_embedding, texts)

    def forward(self, source_images, target_texts, training_mode="diffusion",
                theta_0=None, target_images=None, target_theta=None, source_rendered=None,
                target_alignment=None, source_alignment=None):
        if theta_0 is None:
            theta_0, estimated_alignment = self.get_3d_params(source_images, return_alignment=True)
            if source_alignment is None:
                source_alignment = estimated_alignment
        if training_mode == "3d_editor":
            return {"delta_theta": self.predict_delta(theta_0, target_texts, source_rendered),
                    "theta_0": theta_0}
        if training_mode != "diffusion":
            raise ValueError(f"Unknown training mode: {training_mode}")
        if target_images is None or target_theta is None:
            raise ValueError("Diffusion training requires real paired target images and 3D coefficients")

        if (self.config.lambda_3d or self.config.lambda_id or self.config.lambda_control) and target_alignment is None:
            raise ValueError("3D consistency requires the target portrait's cached face alignment")
        if (self.config.lambda_id or self.config.lambda_control or self.config.lambda_lighting_exposure) and source_alignment is None:
            raise ValueError("Identity supervision requires the source portrait's cached face alignment")

        self._init_diffusion_modules()
        self.initialize_identity_loss()
        diffusion = self.diffusion_unet
        if diffusion.geometry_adapter is not None and source_alignment is None:
            raise ValueError("Spatial geometry requires the source portrait's cached face alignment")
        with torch.no_grad():
            delta = self.predict_delta(theta_0, target_texts, source_rendered)
            geometry = None
            if diffusion.geometry_adapter is not None:
                # Paired renders teach the spatial correspondence. Predicted renders
                # expose the denoiser to the editor errors it will see at inference.
                if self.training and self.config.measured_geometry_probability > 0:
                    editor = self.text_3d_editor
                    mask = editor.attribute_mask([editor.parse_attributes(text) for text in target_texts], theta_0)
                    measured = (target_theta - theta_0).masked_fill(~mask, 0)
                    use_measured = torch.rand(len(delta), 1, device=delta.device)
                    delta = torch.where(use_measured < self.config.measured_geometry_probability, measured, delta)
                geometry = align_geometry(
                    self.render_from_params(theta_0 + delta, geometry=True),
                    source_alignment, source_images.shape[-2:],
                )
            source_latents = self.encode_source_images(source_images)
            source_identity = self.encode_source_identity(source_images)
            target_latents = diffusion.encode_image(target_images)
            dropped = torch.rand(len(delta), device=delta.device) < self.config.cfg_dropout_prob
            if not self.training:
                dropped.fill_(False)
            prompts = ["" if drop else text for text, drop in zip(target_texts, dropped.tolist())]
            text_embeddings = diffusion.encode_prompt(prompts)
        condition = self.diffusion_mapper(
            delta.masked_fill(dropped[:, None], 0), source_latents, text_embeddings
        )
        noise = torch.randn_like(target_latents)
        timesteps = torch.randint(self.config.num_train_timesteps, (len(delta),), device=delta.device)
        source_anchor = ((torch.rand(len(delta), device=delta.device) < self.config.source_edit_probability)
                         & ~dropped)
        if source_anchor.any():
            edit_timesteps = torch.randint(
                self.config.source_edit_min_timestep, self.config.source_edit_max_timestep,
                timesteps.shape, device=timesteps.device,
            )
            timesteps = torch.where(source_anchor, edit_timesteps, timesteps)
        scheduler = diffusion.noise_scheduler
        noisy, effective_noise = make_edit_training_pair(
            scheduler, target_latents, source_latents, noise, timesteps, source_anchor,
        )
        geometry_residuals = diffusion.geometry_features(geometry, noisy.shape[-2:])
        identity_kwargs = {}
        if source_identity is not None:
            identity_kwargs["identity_embeds"] = source_identity.masked_fill(dropped[:, None, None], 0)
        prediction = diffusion(noisy, timesteps, condition, source_images, source_latents,
                               geometry_residuals=geometry_residuals, **identity_kwargs)
        if scheduler.config.prediction_type == "epsilon":
            denoising_target = effective_noise
        elif scheduler.config.prediction_type == "v_prediction":
            denoising_target = scheduler.get_velocity(target_latents, effective_noise, timesteps)
        else:
            denoising_target = target_latents
        loss_diffusion = F.mse_loss(prediction.float(), denoising_target.float())

        alpha = scheduler.alphas_cumprod.to(noisy.device)[timesteps, None, None, None]
        if scheduler.config.prediction_type == "epsilon":
            predicted_clean = (noisy - (1 - alpha).sqrt() * prediction) / alpha.sqrt().clamp_min(1e-4)
        elif scheduler.config.prediction_type == "v_prediction":
            predicted_clean = alpha.sqrt() * noisy - (1 - alpha).sqrt() * prediction
        else:
            predicted_clean = prediction.float()
        loss = self.config.lambda_diffusion * loss_diffusion
        losses = {"loss_diffusion": loss_diffusion,
                  "source_edit_fraction": source_anchor.float().mean()}
        mid_noise = (timesteps >= 100) & (timesteps < self.config.auxiliary_max_timestep)
        if self.config.lambda_diffusion_edit and mid_noise.any():
            edit_loss = F.mse_loss(predicted_clean[mid_noise], target_latents[mid_noise])
            loss = loss + self.config.lambda_diffusion_edit * edit_loss
            losses["loss_diffusion_edit"] = edit_loss

        editable = ((timesteps < self.config.auxiliary_max_timestep) & ~dropped).nonzero(as_tuple=True)[0]
        if len(editable) and (self.config.lambda_clip or self.config.lambda_3d or self.config.lambda_id or self.config.lambda_control or self.config.lambda_lighting_exposure):
            indices = editable[:self.config.max_clip_grad_samples]
            predicted_images = diffusion.decode_latent_train(predicted_clean[indices])
            if getattr(self, "photometric_head", None) is not None:
                predicted_images = self.calibrate_exposure(
                    source_images[indices], predicted_images, [target_texts[i] for i in indices.tolist()],
                    text_embeddings[indices], source_alignment[indices],
                )
            real_images = target_images[indices]
            if self.config.lambda_lighting_exposure:
                attributes = [self.text_3d_editor.parse_attributes(target_texts[i]) for i in indices.tolist()]
                exposure_count = sum("lighting" in requested for requested in attributes)
                if exposure_count:
                    exposure_loss = requested_lighting_exposure_loss(
                        predicted_images, source_images[indices], real_images,
                        source_alignment[indices], attributes,
                    )
                    loss = loss + self.config.lambda_lighting_exposure * exposure_loss
                    losses["loss_lighting_exposure"] = exposure_loss
                    losses["lighting_exposure_sample_count"] = exposure_count
            if self.identity_loss is not None:
                identity_loss = self.identity_loss(
                    predicted_images, source_images[indices], real_images,
                    source_alignment[indices], target_alignment[indices],
                )
                loss = loss + self.config.lambda_id * identity_loss
                losses["loss_identity"] = identity_loss
            if self.config.lambda_clip:
                with torch.no_grad():
                    target_embedding = self.clip_encoder.encode_image(real_images)
                pred_embedding = self.clip_encoder.encode_image(predicted_images)
                clip_loss = 1 - F.cosine_similarity(pred_embedding.float(), target_embedding.float()).mean()
                loss = loss + self.config.lambda_clip * clip_loss
                losses["loss_clip"] = clip_loss
            if self.config.lambda_3d or self.config.lambda_control:
                # Use the same spatial frame for predicted/target images. Deep3D's
                # frozen network remains differentiable with respect to generated pixels.
                pred_coeff = self.deep3d_processor.reconstruct_aligned(
                    predicted_images, target_alignment[indices]
                )
                with torch.no_grad():
                    real_coeff = self.deep3d_processor.reconstruct_aligned(
                        real_images, target_alignment[indices]
                    )
                if self.config.lambda_3d:
                    errors = []
                    for name, group in self.text_3d_editor.attribute_slices.items():
                        if name == "lighting" and getattr(self, "photometric_head", None) is not None:
                            continue
                        scale = 3.0 if name == "pose" and geometry is not None else 1.0
                        errors.append(F.smooth_l1_loss(pred_coeff[:, group] * scale, real_coeff[:, group] * scale))
                    geometry_loss = torch.stack(errors).mean()
                    loss = loss + self.config.lambda_3d * geometry_loss
                    losses["loss_3d"] = geometry_loss
                if self.config.lambda_control:
                    with torch.no_grad():
                        source_coeff = self.deep3d_processor.reconstruct_aligned(
                            source_images[indices], source_alignment[indices])
                    attributes = [self.text_3d_editor.parse_attributes(target_texts[i]) for i in indices.tolist()]
                    if getattr(self, "photometric_head", None) is not None:
                        # Deep3D can absorb exposure into albedo. The explicit
                        # pixel exposure objective supervises global light gain.
                        attributes = [[name for name in row if name != "lighting"] for row in attributes]
                    rows = [i for i, row in enumerate(attributes) if row]
                    if rows:
                        control_loss = requested_attribute_control_loss(
                            pred_coeff[rows], source_coeff[rows], real_coeff[rows],
                            [attributes[i] for i in rows], self.text_3d_editor.attribute_slices,
                        )
                        loss = loss + self.config.lambda_control * control_loss
                        losses["loss_control"] = control_loss

        losses["loss"] = loss
        losses["theta_target"] = theta_0 + delta
        return losses

    def forward_identity_rollout(self, source_images, target_texts, *, theta_0, target_images,
                                 source_rendered, source_alignment, target_alignment,
                                 generator=None):
        """Calibrate bounded identity gains on the actual deployed trajectory."""
        self._init_diffusion_modules()
        self.initialize_identity_loss()
        diffusion = self.diffusion_unet
        # Match generation's FP32 frozen conditioning, even inside the trainer's AMP context.
        with torch.no_grad(), torch.amp.autocast(device_type=source_images.device.type, enabled=False):
            delta = self.predict_delta(theta_0, target_texts, source_rendered)
            source_latents = self.encode_source_images(source_images)
            identity = self.encode_source_identity(source_images)
            tokens = diffusion.encode_prompt(target_texts)
            condition = self.diffusion_mapper(delta, source_latents, tokens)
            geometry = None
            if diffusion.geometry_adapter is not None:
                geometry = align_geometry(self.render_from_params(theta_0 + delta, geometry=True),
                                          source_alignment, source_images.shape[-2:])
        latents = diffusion.sample_train(
            condition, source_images, source_latents, geometry=geometry, identity_embeds=identity,
            guidance_scale=self.config.guidance_scale, num_inference_steps=self.config.num_inference_steps,
            strength=self.config.inference_strength, generator=generator,
        )
        images = diffusion.decode_latent_train(latents)
        images = self.calibrate_exposure(source_images, images, target_texts, tokens, source_alignment)
        identity_loss = self.identity_loss(images, source_images, target_images,
                                           source_alignment, target_alignment)
        loss = self.config.lambda_id * identity_loss
        losses = {"loss_final_identity": identity_loss}
        if hasattr(self.identity_loss, "last_detection_fallbacks"):
            losses["identity_detection_fallbacks"] = self.identity_loss.last_detection_fallbacks
        attributes = [self.text_3d_editor.parse_attributes(text) for text in target_texts]
        if self.config.lambda_lighting_exposure and any("lighting" in row for row in attributes):
            exposure = requested_lighting_exposure_loss(
                images, source_images, target_images, source_alignment, attributes)
            loss = loss + self.config.lambda_lighting_exposure * exposure
            losses["loss_final_exposure"] = exposure
        controlled = [[name for name in row if name != "lighting"] for row in attributes]
        rows = [i for i, row in enumerate(controlled) if row]
        if self.config.lambda_control and rows:
            predicted = self.deep3d_processor.reconstruct_aligned(images[rows], target_alignment[rows])
            with torch.no_grad():
                target = self.deep3d_processor.reconstruct_aligned(target_images[rows], target_alignment[rows])
                source = self.deep3d_processor.reconstruct_aligned(source_images[rows], source_alignment[rows])
            control = requested_attribute_control_loss(predicted, source, target,
                        [controlled[i] for i in rows], self.text_3d_editor.attribute_slices)
            loss = loss + self.config.lambda_control * control
            losses["loss_final_control"] = control
        gains = [p for name, p in self.named_parameters() if name.endswith(".identity_gain_logit")]
        regularizer = torch.stack([p.square() for p in gains]).mean()
        losses["loss_gain_regularizer"] = regularizer
        losses["loss"] = loss + .001 * regularizer
        return losses

    @torch.no_grad()
    def generate(self, source_images, target_texts, guidance_scale=7.5, num_inference_steps=50,
                 *, strength=None, generator=None, theta_0=None, source_rendered=None,
                 estimate_final_theta=True, delta_theta=None, use_geometry=True,
                 source_alignment=None):
        self._init_diffusion_modules()
        diffusion = self.diffusion_unet
        if strength is None:
            strength = self.config.inference_strength
        spatial = use_geometry and diffusion.geometry_adapter is not None
        if (spatial or self.photometric_head is not None) and source_alignment is None:
            estimated_theta, source_alignment = self.get_3d_params(source_images, return_alignment=True)
            if theta_0 is None:
                theta_0 = estimated_theta
        elif theta_0 is None and (delta_theta is None or spatial):
            theta_0 = self.get_3d_params(source_images)
        if delta_theta is None:
            delta_theta = self.predict_delta(theta_0, target_texts, source_rendered)
        source_latents = self.encode_source_images(source_images)
        source_identity = self.encode_source_identity(source_images)
        geometry = None
        if spatial:
            geometry = align_geometry(
                self.render_from_params(theta_0 + delta_theta, geometry=True),
                source_alignment, source_images.shape[-2:],
            )
        text_tokens = diffusion.encode_prompt(target_texts)
        condition = self.diffusion_mapper(delta_theta, source_latents, text_tokens)
        identity_kwargs = {"identity_embeds": source_identity} if source_identity is not None else {}
        generated_latents = diffusion.sample(
            condition, source_images, source_latents, guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps, generator=generator, strength=strength,
            geometry=geometry,
            **identity_kwargs,
        )
        images = diffusion.decode_latent(generated_latents)
        images = self.calibrate_exposure(source_images, images, target_texts, text_tokens, source_alignment)
        if not estimate_final_theta:
            return images, None
        return images, self.get_3d_params(images)

    def train_vae_only(self, source_images):
        """仅训练VAE (如果VAE未冻结)"""
        self._init_diffusion_modules()
        if self.config.freeze_vae:
            print("VAE is frozen, skipping VAE-only training")
            return torch.tensor(0.0, device=self.device)

        vae = self.diffusion_unet.vae
        posterior = vae.encode(source_images).latent_dist
        reconstructed_images = vae.decode(posterior.sample()).sample
        recon_loss = F.mse_loss(reconstructed_images, source_images)
        kl_loss = posterior.kl().mean()
        vae_loss = recon_loss + self.config.vae_kl_weight * kl_loss

        return {
            "loss": vae_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "reconstructed_images": reconstructed_images
        }
