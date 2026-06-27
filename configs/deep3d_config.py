from dataclasses import dataclass
from .base_config import BaseConfig


@dataclass
class Deep3DConfig(BaseConfig):
    """Deep3DFaceReconstruction 专用配置"""

    # Deep3DFaceReconstruction 路径
    deep3d_repo_path: str = "/path/to/Deep3DFaceReconstruction"
    deep3d_checkpoint_path: str = "/path/to/Deep3DFaceReconstruction/checkpoints/face_recon_model.pth"

    # 3DMM 参数配置
    shape_dim: int = 100
    expression_dim: int = 79
    pose_dim: int = 3
    texture_dim: int = 100
    lighting_dim: int = 27

    # 相机参数
    camera_f: float = 1015.0
    camera_c: tuple = (112.0, 112.0)

    # 人脸关键点检测模型路径
    landmark_model_path: str = "/path/to/landmark/model"

    # BFM 模型路径
    bfm_path: str = "/path/to/Deep3DFaceReconstruction/BFM/BFM_model_front.mat"

    @property
    def total_3d_dim(self):
        return self.shape_dim + self.expression_dim + self.pose_dim + self.texture_dim + self.lighting_dim