from dataclasses import dataclass
from .base_config import BaseConfig
from typing import Optional

@dataclass
class Stage2Config(BaseConfig):
    # 训练参数
    batch_size: int = 1
    num_epochs: int = 100
    learning_rate: float = 1e-4
    unet_learning_rate: float = 1e-5  # protect pretrained attention; adapters use learning_rate
    weight_decay: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.999

    # 扩散模型参数
    diffusion_model_name: str = "runwayml/stable-diffusion-v1-5"
    diffusion_local_path: str = ""  # 离线时设为本地文件夹路径，如 "./models/stable-diffusion-v1-5"
    num_train_timesteps: int = 1000
    beta_start: float = 0.00085
    beta_end: float = 0.012
    beta_schedule: str = "scaled_linear"
    prediction_type: str = "epsilon"  # 扩散模型预测类型: "epsilon" 或 "sample"

    # 损失权重
    lambda_diffusion: float = 1.0   # DDPM 基础去噪损失 (有梯度)
    lambda_diffusion_edit: float = 0.05  # x0 auxiliary loss against paired target latent
    lambda_3d: float = 0.1          # differentiable paired 3D consistency
    lambda_control: float = 0.0  # normalized requested-attribute error; copy baseline ~1
    lambda_lighting_exposure: float = 0.0  # requested face brightness in pixel space
    source_edit_probability: float = 0.0  # opt-in supervised source-start transport
    source_edit_min_timestep: int = 250
    source_edit_max_timestep: int = 700
    auxiliary_max_timestep: int = 500
    lambda_id: float = 0.0          # frozen FaceNet identity supervision on generated crops
    identity_weights_path: str = ""
    identity_loss_backend: str = "facenet"
    identity_arcface_weights_path: str = ""
    identity_training_alignment: str = "generated"
    identity_exposure_normalization: bool = False
    faceid_adapter_path: str = ""
    faceid_encoder_path: str = ""
    faceid_scale: float = 1.0
    isolate_faceid_attention: bool = False
    identity_gain_only: bool = False
    identity_refinement_rank: int = 0
    identity_rollout_steps_per_epoch: int = 32
    identity_calibration_patience: int = 5
    identity_validation_subjects: int = 0
    use_photometric_head: bool = False
    use_geometry_adapter: bool = False
    measured_geometry_probability: float = 0.5
    use_bf16: bool = False
    lambda_clip: float = 0.05       # weak semantic auxiliary; do not overpower paired supervision
    lambda_vae: float = 0.0         # VAE 损失 (VAE 冻结时保持为 0)
    lambda_diversity: float = 0.0   # unsafe without explicit contrastive pairs
    vae_kl_weight: float = 1e-6     # VAE KL 散度权重

    # ── 图像质量评估指标 ──
    compute_metrics: bool = True           # 主开关: 是否计算图像质量指标
    ssim_enabled: bool = True              # SSIM (生成图与目标图的结构相似性)
    psnr_enabled: bool = True              # PSNR (峰值信噪比) — 衡量像素级保真度, 值越高越好
    lpips_enabled: bool = False            # LPIPS (感知相似度) — 需要 pip install lpips, 值越低越好
    fid_enabled: bool = False               # FID (分布距离) — 衡量生成图像真实性, 值越低越好
    fid_num_samples: int = 100             # 单次 FID 评估使用的样本数
    fid_eval_every_n_epochs: int = 5       # 每 N 个 epoch 计算一次 FID

    # 推理参数
    guidance_scale: float = 7.5
    num_inference_steps: int = 50
    inference_strength: float = 1.0  # legacy default; repair runs explicitly use img2img

    # 模型参数
    condition_seq_length: int = 77
    condition_embed_dim: int = 768

    # 日志和保存
    log_interval: int = 100
    save_interval: int = 1000
    eval_interval: int = 0  # full validation is run once per epoch

    # 恢复训练
    resume_from_checkpoint: Optional[str] = None
    stage1_checkpoint_path: Optional[str] = None  # 第一阶段训练好的模型

    def __post_init__(self):
        if self.identity_training_alignment not in {"generated", "paired_target"}:
            raise ValueError("Unknown identity training alignment")
        if self.identity_loss_backend not in {"facenet", "arcface"}:
            raise ValueError("Unknown identity loss backend")
        if self.identity_refinement_rank < 0 or (self.identity_refinement_rank and not self.identity_gain_only):
            raise ValueError("Identity refinement requires identity calibration mode and a nonnegative rank")
        if self.identity_loss_backend == "arcface":
            if not self.identity_arcface_weights_path:
                raise ValueError("ArcFace identity loss requires its ONNX weights")
            if self.identity_exposure_normalization:
                raise ValueError("ArcFace identity loss uses raw exposure normalization")
        if self.faceid_adapter_path and not self.faceid_encoder_path:
            raise ValueError("FaceID requires the matching buffalo_l encoder")
        if self.faceid_scale < 0:
            raise ValueError("faceid_scale must be nonnegative")
        if self.isolate_faceid_attention and not self.faceid_adapter_path:
            raise ValueError("Isolated identity attention requires a FaceID adapter")
        if self.identity_gain_only and not self.isolate_faceid_attention:
            raise ValueError("Identity calibration requires isolated FaceID attention")
        if self.identity_gain_only and self.lambda_id <= 0:
            raise ValueError("Identity calibration requires a positive identity loss weight")
        if self.identity_rollout_steps_per_epoch < 1:
            raise ValueError("Identity rollout steps must be positive")
        if self.identity_calibration_patience < 1:
            raise ValueError("Identity calibration patience must be positive")
        if self.identity_validation_subjects < 0:
            raise ValueError("Identity validation subjects must be nonnegative")
        if not 0 <= self.source_edit_probability <= 1:
            raise ValueError("source_edit_probability must be in [0, 1]")
        if not 0 <= self.source_edit_min_timestep < self.source_edit_max_timestep <= self.num_train_timesteps:
            raise ValueError("Invalid source edit timestep range")
        if not 0 < self.auxiliary_max_timestep <= self.num_train_timesteps:
            raise ValueError("Invalid auxiliary_max_timestep")
        if self.lambda_control < 0 or self.lambda_lighting_exposure < 0:
            raise ValueError("Control loss weights must be nonnegative")
        if not 0 < self.inference_strength <= 1:
            raise ValueError("inference_strength must be in (0, 1]")
        if self.unet_learning_rate <= 0:
            raise ValueError("unet_learning_rate must be positive")
        if not 0 <= self.measured_geometry_probability <= 1:
            raise ValueError("measured_geometry_probability must be in [0, 1]")
        super().__post_init__()
