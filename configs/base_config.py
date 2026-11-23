from dataclasses import dataclass
from typing import Optional, List
import os
import torch
#abc

@dataclass
class BaseConfig:
    # 设备设置
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

    # 数据设置
    data_root: str = "./datasets/face_editing"
    # SD 1.5 was trained at 512px. 224px puts its VAE/UNet far outside the
    # training distribution and is not suitable for photo-quality editing.
    image_size: int = 512
    batch_size: int = 32
    num_workers: int = 8
    balance_instructions: bool = False
    augment_instructions: bool = False
    align_paired_targets: bool = False

    # 模型路径
    clip_model_name: str = "openai/clip-vit-large-patch14"
    clip_local_path: str = ""  # 离线时设为本地文件夹路径，如 "./models/clip-vit-large-patch14"
    project_root: str = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    deep3d_repo_path: str = ""
    deep3d_checkpoint_path: str = ""
    face_detector_path: str = ""
    deep3d_model_path: str = "/path/to/deep3d/model"
    arcface_model_path: str = "/path/to/arcface/model"

    # BFM 模型路径 (用于 PyTorch3D 渲染)
    bfm_model_path: str = "/path/to/BFM/model"

    # 3DMM 参数维度 (匹配 Deep3DFaceReconstruction 的 split_coeff 输出)
    # id:80, exp:64, tex:80, angle:3, gamma:27 (不含 trans/camera)
    shape_dim: int = 80
    expression_dim: int = 64
    pose_dim: int = 3  # 旋转角度 (pitch, yaw, roll)
    texture_dim: int = 80
    lighting_dim: int = 27  # 球谐光照系数

    # 相机参数
    camera_f: float = 1015.0  # 焦距
    camera_c: tuple = (112.0, 112.0)  # 相机中心

    # 渲染参数
    render_image_size: int = 224  # 渲染分辨率 (匹配 CLIP 224×224，避免上采样丢失细节)

    # CLIP 模型参数 (Stage1/Stage2 共用)
    clip_embed_dim: int = 768  # CLIP ViT-L/14 joint embedding space dimension

    # 多模态输入和冻结设置
    use_multimodal_input: bool = True  # source latent/RGB are required to preserve identity
    freeze_vae: bool = True  # 是否冻结 Stable Diffusion VAE

    # 扩散模型参数 (Stage1/Stage2 共用)
    diffusion_model_name: str = "runwayml/stable-diffusion-v1-5"
    diffusion_local_path: str = ""  # 离线时设为本地文件夹路径
    num_train_timesteps: int = 1000
    beta_start: float = 0.00085
    beta_end: float = 0.012
    beta_schedule: str = "scaled_linear"
    prediction_type: str = "epsilon"

    # 条件嵌入参数 (ConditionalDiffusionMappingNetwork 使用)
    condition_seq_length: int = 77
    condition_embed_dim: int = 768

    # 训练稳定性参数
    max_grad_norm: float = 1.0  # 梯度裁剪阈值
    delta_theta_l2_weight: float = 0.005  # delta_theta L2 正则化权重（提升以抑制Δθ膨胀）
    delta_theta_clamp: float = 2.0  # delta_theta 裁剪范围 [-clamp, clamp]（收紧以减少渲染NaN）
    max_clip_grad_samples: int = 1  # loss_clip 梯度链路每 batch 最多处理的样本数（限制 CLIP 显存）
    cfg_dropout_prob: float = 0.1  # train the unconditional branch used by CFG at inference

    @property
    def total_3d_dim(self):
        return self.shape_dim + self.expression_dim + self.pose_dim + self.texture_dim + self.lighting_dim

    # 输出设置
    output_dir: str = "./outputs"
    log_dir: str = "./logs"
    checkpoint_dir: str = "./checkpoints"

    def __post_init__(self):
        if not self.deep3d_repo_path:
            self.deep3d_repo_path = os.path.join(self.project_root, "Deep3DFR")
        if not self.deep3d_checkpoint_path:
            self.deep3d_checkpoint_path = os.path.join(
                self.deep3d_repo_path,
                "checkpoints",
                "face_recon_model.pth"
            )
        if not self.bfm_model_path or self.bfm_model_path == "/path/to/BFM/model":
            self.bfm_model_path = os.path.join(
                self.deep3d_repo_path,
                "BFM",
                "BFM_model_front.mat"
            )
        if not self.face_detector_path:
            self.face_detector_path = os.path.join(
                self.project_root, "models", "face_detection_yunet_2023mar.onnx"
            )

        # 创建必要的目录
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
