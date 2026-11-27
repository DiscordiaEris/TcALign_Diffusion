from .base_trainer import BaseTrainer
from collections import defaultdict
import torch
from torch.utils.data import default_collate
import os
import math


class Stage2Trainer(BaseTrainer):
    """第二阶段训练器：训练条件扩散映射网络 - 支持多模态输入"""

    def on_training_start(self):
        self.initialize_identity_validation()

    def initialize_identity_validation(self):
        if not getattr(self.config, "identity_validation_subjects", 0):
            return
        gain_only = getattr(self.config, "identity_gain_only", False)
        # latest is saved before epoch-end validation; the on-disk best can
        # therefore carry a newer, higher score after an interrupted epoch.
        directory = getattr(self.config, "checkpoint_dir", None)
        if directory:
            path = os.path.join(directory, "best_identity_checkpoint.pth")
            if os.path.isfile(path):
                saved = torch.load(path, map_location="cpu", weights_only=True)
                score = saved.get("identity_validation", {}).get("best_score")
                current_score = getattr(self, "_best_sampled_identity_score", None)
                if current_score is None or not math.isfinite(float(current_score)):
                    current_score = -1.0
                if score is not None and math.isfinite(float(score)):
                    self._best_sampled_identity_score = max(
                        current_score, score)
                calibration = saved.get("identity_validation", {}).get("calibration", {})
                if gain_only:
                    for key in ("expression_baseline", "max_expression_error",
                                "baseline_progress", "min_expression_progress", "min_arcface_score"):
                        if key in calibration and not hasattr(self, "_identity_calibration_" + key):
                            setattr(self, "_identity_calibration_" + key, calibration[key])
                del saved
        current_score = getattr(self, "_best_sampled_identity_score", None)
        if (current_score is not None and math.isfinite(float(current_score))
                and (not gain_only or hasattr(self, "_identity_calibration_min_expression_progress"))
                and (getattr(self.config, "identity_loss_backend", "facenet") != "arcface"
                     or hasattr(self, "_identity_calibration_min_arcface_score"))):
            return
        device = torch.device(self.device)
        devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
        was_training = self.model.training
        self._initializing_identity_validation = True
        try:
            self.model.eval()
            with torch.random.fork_rng(devices=devices):
                self.validate_sampled_identity()
        finally:
            self._initializing_identity_validation = False
            self.model.train(was_training)

    def _optimizer_learning_rate(self, parameter_name):
        if parameter_name.rsplit(".", 1)[-1] == "identity_gain_logit" or ".identity_refinement." in parameter_name:
            return self.config.learning_rate
        if (parameter_name.startswith("diffusion_unet.unet.")
                and not parameter_name.startswith("diffusion_unet.unet.conv_in.")):
            # Preserve the pretrained denoising prior while new source/geometry
            # paths learn at the full adapter learning rate.
            return getattr(self.config, "unet_learning_rate", self.config.learning_rate * 0.1)
        return self.config.learning_rate

    def __init__(self, config, model, train_dataloader, val_dataloader=None):
        super().__init__(config, model, train_dataloader, val_dataloader)
        self.num_epochs = config.num_epochs

        # 加载第一阶段训练好的模型
        if hasattr(config, 'stage1_checkpoint_path') and config.stage1_checkpoint_path:
            checkpoint = torch.load(config.stage1_checkpoint_path, map_location="cpu", weights_only=True)
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
                self.model.text_3d_editor.load_state_dict(full_state_dict, strict=True)
                self.logger.info(f"Loaded stage1 checkpoint (strict=True) from {config.stage1_checkpoint_path}")

        # 冻结第一阶段模型
        for param in self.model.text_3d_editor.parameters():
            param.requires_grad = False

        # 提前初始化扩散模块并重建优化器（惰性加载在 super().__init__ 之后才触发）
        self.model._init_diffusion_modules()
        self.model.initialize_identity_loss()
        if self.model.diffusion_unet.geometry_adapter is not None:
            self.logger.info("Target geometry adapter enabled; measured training geometry probability %.2f",
                             config.measured_geometry_probability)
        if self.model.identity_loss is not None:
            self.logger.info("Frozen %s identity loss enabled with weight %.3f",
                             getattr(config, "identity_loss_backend", "facenet"), config.lambda_id)
        self.optimizer = self.configure_optimizers()
        self.logger.info("Optimizer re-created with diffusion modules included")
        if getattr(config, "identity_gain_only", False):
            active = [(name, parameter) for name, parameter in self.model.named_parameters()
                      if parameter.requires_grad]
            refinement = getattr(config, "identity_refinement_rank", 0) > 0
            if not active or any(not (".identity_refinement." in name if refinement else
                                     name.endswith(".identity_gain_logit")) for name, _ in active):
                raise RuntimeError("Identity calibration must only update the selected identity branch")
            self.logger.info("Identity calibration: %d identity parameters trainable, frozen generator in eval mode; "
                             "%d full-sampling updates per epoch, learning rate %.6f",
                             sum(parameter.numel() for _, parameter in active),
                             config.identity_rollout_steps_per_epoch, config.learning_rate)
        else:
            self.logger.info("Edit training: source_start_probability=%.2f, timestep_range=[%d,%d), "
                             "control_weight=%.2f, exposure_weight=%.2f, registered_targets=%s",
                             config.source_edit_probability, config.source_edit_min_timestep,
                             config.source_edit_max_timestep, config.lambda_control,
                             config.lambda_lighting_exposure, config.align_paired_targets)


        # ── 初始化图像质量指标计算器 ──
        self.metrics_computer = None
        self._fid_real_features_precomputed = False
        self._last_fid_epoch = -1
        if getattr(config, 'compute_metrics', False):
            from utils.metrics import ImageQualityMetrics
            self.metrics_computer = ImageQualityMetrics(config, self.device)
            self.logger.info("Image quality metrics enabled: "
                             f"SSIM={config.ssim_enabled}, PSNR={config.psnr_enabled}, "
                             f"LPIPS={config.lpips_enabled}, FID={config.fid_enabled}")

    def training_step(self, source_images, target_texts,
                       source_params=None, source_rendered=None, target_images=None,
                       target_params=None, target_alignment=None, source_alignment=None, **kwargs):
        """训练步骤 - 使用预计算参数避免重复跑 Deep3D"""
        if getattr(self.config, "identity_gain_only", False):
            return self.model.forward_identity_rollout(
                source_images, target_texts, theta_0=source_params, target_images=target_images,
                source_rendered=source_rendered, source_alignment=source_alignment,
                target_alignment=target_alignment,
            )
        loss_dict = self.model(
            source_images, target_texts, training_mode="diffusion",
            theta_0=source_params,  # 传入预计算的 3D 参数
            target_images=target_images,
            target_theta=target_params,
            source_rendered=source_rendered,
            target_alignment=target_alignment,
            source_alignment=source_alignment,
        )
        return loss_dict

    def evaluate(self):
        # Validation must compare the same noise/timesteps each epoch, and must
        # not perturb the training RNG or the instruction sampler's generator.
        device = torch.device(self.device)
        devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(self.config.seed)
            if devices:
                with torch.cuda.device(devices[0]):
                    torch.cuda.manual_seed(self.config.seed)
            return self._evaluate_fixed_noise()

    def _evaluate_fixed_noise(self):
        """评估 - 使用预计算参数避免 Deep3D"""
        if getattr(self.config, "identity_gain_only", False):
            was_training = self.model.training
            try:
                self.model.eval()
                return self.validate_sampled_identity()
            finally:
                self.model.train(was_training)
        self.model.eval()
        total_loss = 0
        num_batches = 0
        component_sums = defaultdict(float)
        component_counts = defaultdict(int)

        try:
            with torch.no_grad():
                for batch in self.val_dataloader:
                    source_images = batch["image"].to(self.device)
                    target_texts = batch["text"]
                    source_params = batch.get("params")
                    target_images = batch.get("target_image")
                    target_params = batch.get("target_params")
                    if source_params is not None:
                        source_params = source_params.to(self.device)
                    if target_images is not None:
                        target_images = target_images.to(self.device)
                    if target_params is not None:
                        target_params = target_params.to(self.device)

                    # 计算损失 (传入预计算的 3D 参数跳过 Deep3D)
                    loss_dict = self.model(
                        source_images, target_texts, training_mode="diffusion",
                        theta_0=source_params,
                        target_images=target_images,
                        target_theta=target_params,
                        source_rendered=batch["rendered_image"].to(self.device),
                        target_alignment=batch["target_alignment"].to(self.device),
                        source_alignment=batch["source_alignment"].to(self.device),
                    )
                    total_loss += loss_dict['loss'].item()
                    num_batches += 1
                    for key in ("loss_diffusion", "loss_diffusion_edit", "loss_identity", "loss_3d", "loss_clip", "loss_control", "source_edit_fraction", "loss_lighting_exposure"):
                        if key in loss_dict:
                            count = (int(loss_dict["lighting_exposure_sample_count"])
                                     if key == "loss_lighting_exposure" else 1)
                            component_sums[key] += float(loss_dict[key]) * count
                            component_counts[key] += count

            if num_batches > 0:
                avg_loss = total_loss / num_batches
                self.logger.info(f"Validation loss: {avg_loss:.4f}")
                for key, total in component_sums.items():
                    count = component_counts[key]
                    self.logger.info("Validation %s: %.4f (%d samples/batches with auxiliary loss)",
                                     key, total / count, count)
                # 生成示例图像
                self.generate_examples()
                if getattr(self.config, "identity_validation_subjects", 0):
                    self.validate_sampled_identity()

                # ── FID 评估 (仅在 epoch 结束时按间隔运行) ──
                if (self.metrics_computer is not None
                        and getattr(self.config, 'fid_enabled', False)
                        and self.epoch % self.config.fid_eval_every_n_epochs == 0
                        and self.epoch != self._last_fid_epoch
                        and self.epoch > 0):
                    self._evaluate_fid_metrics()
                    self._last_fid_epoch = self.epoch
            else:
                avg_loss = 0.0
                self.logger.info("No validation data available, skipping evaluation")
        finally:
            self.model.train()

        return avg_loss

    @torch.no_grad()
    def validate_sampled_identity(self):
        """Retain weights using final-image identity with pose/exposure gates."""
        import json
        from pathlib import Path
        from utils.identity_validation import evaluate_sampled_identity, should_replace_identity_checkpoint

        result = evaluate_sampled_identity(
            self.model, self.val_dataloader.dataset, self.device,
            subjects=self.config.identity_validation_subjects, seed=self.config.seed,
        )
        metrics = result["aggregate"]
        self._last_sampled_identity_metrics = metrics
        path = Path(self.config.output_dir) / f"identity_validation_step_{self.global_step}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        self.logger.info("Final-sample identity validation: %s", metrics)
        previous = getattr(self, "_best_sampled_identity_score", -1.)
        if previous is None or not math.isfinite(float(previous)):
            previous = -1.0
        gain_only = getattr(self.config, "identity_gain_only", False)
        establishing_baseline = gain_only and not hasattr(
            self, "_identity_calibration_min_expression_progress")
        if establishing_baseline:
            try:
                expression_baseline = float(metrics["expression_error_ratio"])
                baseline_progress = float(metrics["expression_progress"])
            except (KeyError, TypeError, ValueError):
                expression_baseline = math.nan
                baseline_progress = math.nan
            if (metrics.get("expression_expected") != 6 or metrics.get("expression_valid") != 6
                    or metrics.get("expression_progress_valid") != 6
                    or metrics.get("expression_direction_hits") != 6
                    or not math.isfinite(expression_baseline) or expression_baseline < 0
                    or not math.isfinite(baseline_progress) or not 0.1 <= baseline_progress <= 1):
                raise RuntimeError("Identity calibration requires a finite six-subject expression baseline with six valid directions")
            self._identity_calibration_expression_baseline = expression_baseline
            self._identity_calibration_max_expression_error = expression_baseline * 1.1
            self._identity_calibration_baseline_progress = baseline_progress
            self._identity_calibration_min_expression_progress = max(0.1, baseline_progress * 0.9)
            self._identity_calibration_checks_without_improvement = 0
        if gain_only and not hasattr(self, "_identity_calibration_patience"):
            self._identity_calibration_patience = getattr(self.config, "identity_calibration_patience", 5)
        gate_options = ({"require_expression": True,
                         "max_expression_error": self._identity_calibration_max_expression_error,
                         "min_expression_progress": self._identity_calibration_min_expression_progress}
                        if gain_only else {})
        establishing_arcface_baseline = False
        if gain_only and getattr(self.config, "identity_loss_backend", "facenet") == "arcface":
            if not hasattr(self, "_identity_calibration_min_arcface_score"):
                value = metrics.get("arcface_raw_cosine")
                if metrics.get("arcface_raw_valid") != 18 or value is None or not math.isfinite(float(value)):
                    raise RuntimeError("Identity calibration requires a complete ArcFace identity baseline")
                self._identity_calibration_min_arcface_score = value
                establishing_arcface_baseline = True
            gate_options["min_arcface_score"] = self._identity_calibration_min_arcface_score
        improved = should_replace_identity_checkpoint(metrics, previous, **gate_options)
        initializing = (establishing_baseline or establishing_arcface_baseline
                        or getattr(self, "_initializing_identity_validation", False))
        if gain_only:
            if improved:
                self._identity_calibration_checks_without_improvement = 0
                self.stop_training = False
            elif (not initializing
                  and getattr(self, "_identity_calibration_last_validation_epoch", None) != self.epoch):
                self._identity_calibration_checks_without_improvement = getattr(
                    self, "_identity_calibration_checks_without_improvement", 0) + 1
            if not initializing:
                self._identity_calibration_last_validation_epoch = self.epoch
            if (getattr(self, "_identity_calibration_checks_without_improvement", 0)
                    >= self._identity_calibration_patience):
                self.stop_training = True
                self.logger.info("Stopping identity calibration after %d epochs without sampled identity improvement",
                                 self._identity_calibration_checks_without_improvement)
        if improved:
            self._best_sampled_identity_score = metrics["identity_score"]
            checkpoint = os.path.join(self.config.checkpoint_dir, "best_identity_checkpoint.pth")
            self.save_checkpoint(checkpoint)
            self.logger.info("Best sampled identity updated: %.4f -> %.4f", previous,
                             self._best_sampled_identity_score)
        if gain_only:
            if not hasattr(self, "_best_sampled_identity_score"):
                self._best_sampled_identity_score = previous
            self.save_checkpoint(os.path.join(self.config.checkpoint_dir, "latest_checkpoint.pth"))
        return metrics

    @torch.no_grad()
    def _precompute_real_features_for_fid(self):
        """预计算验证集源图像的 InceptionV3 特征 (FID 基准, 只运行一次)"""
        if self._fid_real_features_precomputed or self.metrics_computer is None:
            return

        self.logger.info("Precomputing InceptionV3 features for FID reference distribution...")
        self.metrics_computer.reset_fid_state()
        for batch in self.val_dataloader:
            target_images = batch["target_image"].to(self.device)
            self.metrics_computer.accumulate_real_features(target_images)

        self._fid_real_features_precomputed = True
        self.logger.info("FID reference features precomputed successfully")

    @torch.no_grad()
    def _evaluate_fid_metrics(self):
        """生成 N 张图像并计算 FID 指标"""
        # 确保真实特征已预计算
        self._precompute_real_features_for_fid()

        num_samples = self.config.fid_num_samples
        collected = 0
        gen_features_accumulated = []  # type: list[torch.Tensor]

        self.logger.info(f"Evaluating FID: generating {num_samples} images...")

        for batch in self.val_dataloader:
            if collected >= num_samples:
                break

            source_images = batch["image"].to(self.device)
            target_texts = batch["text"]

            # 取当前 batch 需要的样本数
            take = min(source_images.shape[0], num_samples - collected)
            src = source_images[:take]
            txt = target_texts[:take]

            generated, _ = self.model.generate(
                src, txt,
                guidance_scale=self.config.guidance_scale,
                num_inference_steps=self.config.num_inference_steps,
                theta_0=batch["params"][:take].to(self.device),
                source_alignment=batch["source_alignment"][:take].to(self.device),
                source_rendered=batch["rendered_image"][:take].to(self.device),
                estimate_final_theta=False,
            )

            gen_features_accumulated.append(generated)
            collected += take

        # 拼接生成图像并提取特征
        gen_cat = torch.cat(gen_features_accumulated, dim=0)

        self.metrics_computer.reset_fid_state()
        # 重新累积真实特征 (因为之前 compute_fid 或 reset 可能清空了)
        for batch in self.val_dataloader:
            target_images = batch["target_image"].to(self.device)
            self.metrics_computer.accumulate_real_features(target_images)
        self.metrics_computer.accumulate_gen_features(gen_cat)

        try:
            fid_score = self.metrics_computer.compute_fid()
            self.logger.info(f"FID (epoch {self.epoch}, {num_samples} samples): {fid_score:.4f}")
        except Exception as e:
            self.logger.warning(f"FID computation failed: {e}")
        finally:
            self.metrics_computer.reset_fid_state()
            # 重置预计算标志, 下次 FID 评估时重新提取真实特征
            self._fid_real_features_precomputed = False

    def generate_examples(self):
        """生成示例图像"""
        dataset = self.val_dataloader.dataset
        previews = {}
        for index, (_, _, text) in enumerate(dataset.metadata):
            group = "+".join(self.model.text_3d_editor.parse_attributes(text))
            if group == "pose":
                if "left side of the image" in text.lower():
                    group = "left pose"
                elif "right side of the image" in text.lower():
                    group = "right pose"
            previews.setdefault(group, index)
        priority = ("left pose", "right pose", "expression", "lighting")
        selected = sorted(previews, key=lambda group: priority.index(group) if group in priority else len(priority))
        batch = default_collate([dataset[previews[group]] for group in selected[:4]])
        source_images = batch["image"].to(self.device)
        target_images = batch["target_image"].to(self.device)
        target_texts = batch["text"]
        generator = torch.Generator(device=self.device).manual_seed(self.config.seed)

        # 生成图像
        with torch.no_grad():
            generated_images, _ = self.model.generate(
                source_images, target_texts,
                guidance_scale=self.config.guidance_scale,
                num_inference_steps=self.config.num_inference_steps,
                theta_0=batch["params"].to(self.device),
                source_alignment=batch["source_alignment"].to(self.device),
                source_rendered=batch["rendered_image"].to(self.device),
                generator=generator,
                estimate_final_theta=False,
            )

        # 保存图像
        from utils.visualization import save_image_grid
        output_path = os.path.join(
            self.config.output_dir,
            f"examples_step_{self.global_step}.png"
        )
        save_image_grid(source_images, generated_images, target_texts, output_path,
                        target_images=target_images)

        self.logger.info(f"Example images saved at {output_path}")

        # ── 计算并记录 per-sample 图像质量指标 ──
        if self.metrics_computer is not None:
            metrics = self.metrics_computer.compute_per_sample_metrics(
                target_images, generated_images
            )
            if metrics:
                metric_log = " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
                self.logger.info(f"Example metrics (step {self.global_step}): {metric_log}")
            copy_metrics = self.metrics_computer.compute_per_sample_metrics(target_images, source_images)
            self.logger.info("Source-copy baseline metrics: %s", copy_metrics)

        # Denoising MSE alone can improve while sampling ignores the instruction.
        # Monitor actual sampled images with independent face features and a
        # fixed aligned 3D estimator, alongside the unchanged-source baseline.
        from utils.editing_metrics import coefficient_edit_scores
        from utils.losses import requested_lighting_exposure_loss
        with torch.no_grad():
            source_alignment = batch["source_alignment"].to(self.device)
            target_alignment = batch["target_alignment"].to(self.device)
            if self.model.identity_loss is not None:
                identity = self.model.identity_loss
                source_features = identity.embed(source_images, source_alignment)
                generated_features = identity.embed(generated_images, target_alignment)
                target_features = identity.embed(target_images, target_alignment)
                generated_cosine = torch.nn.functional.cosine_similarity(source_features, generated_features)
                target_cosine = torch.nn.functional.cosine_similarity(source_features, target_features)
                self.logger.info("Sampled identity cosine: generated=%.4f, paired_target=%.4f; per_edit=%s",
                                 generated_cosine.mean(), target_cosine.mean(), generated_cosine.tolist())
            generated_coeff = self.model.deep3d_processor.reconstruct_aligned(generated_images, target_alignment)
            target_coeff = self.model.deep3d_processor.reconstruct_aligned(target_images, target_alignment)
            source_coeff = self.model.deep3d_processor.reconstruct_aligned(source_images, source_alignment)
            scores = coefficient_edit_scores(
                source_coeff, target_coeff, generated_coeff,
                [self.model.text_3d_editor.parse_attributes(text) for text in target_texts],
                self.model.text_3d_editor.attribute_slices,
            )
            self.logger.info("Sampled edit metrics (fixed aligned estimator): %s", scores)
            attributes = [self.model.text_3d_editor.parse_attributes(text) for text in target_texts]
            if any("lighting" in selected for selected in attributes):
                exposure = requested_lighting_exposure_loss(
                    generated_images, source_images, target_images, source_alignment, attributes)
                self.logger.info("Sampled lighting exposure error: %.4f (copy baseline ~1, target 0)", exposure)
