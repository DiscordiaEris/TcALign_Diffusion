from dataclasses import dataclass
from .base_config import BaseConfig
from typing import Optional

@dataclass
class Stage2Config(BaseConfig):
    # 训练参数
    num_epochs: int = 100
    learning_rate: float = 1e-4
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
    lambda_diffusion: float = 1.0  # 扩散重建损失
    lambda_3d: float = 1.0  # 3D 一致性损失
    lambda_id: float = 0.1  # 身份保持损失
    lambda_clip: float = 0.5  # CLIP 对齐损失
    lambda_vae: float = 0.0  # VAE 损失 (VAE 冻结时设为 0)
    vae_kl_weight: float = 1e-6  # VAE KL 散度权重

    # 推理参数
    guidance_scale: float = 7.5
    num_inference_steps: int = 50

    # 模型参数
    condition_seq_length: int = 77
    condition_embed_dim: int = 768

    # 日志和保存
    log_interval: int = 100
    save_interval: int = 1000
    eval_interval: int = 500

    # 恢复训练
    resume_from_checkpoint: Optional[str] = None
    stage1_checkpoint_path: Optional[str] = None  # 第一阶段训练好的模型