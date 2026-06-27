from dataclasses import dataclass
from .base_config import BaseConfig


@dataclass
class PyTorch3DConfig(BaseConfig):
    """PyTorch3D 渲染专用配置"""

    # BFM 模型路径
    bfm_model_path: str = "/path/to/BFM/model"

    # 渲染参数
    render_image_size: int = 224
    render_fov: float = 12.5936  # BFM 模型的默认视场角

    # 光照参数
    ambient_color: tuple = (0.8, 0.8, 0.8)
    diffuse_color: tuple = (0.2, 0.2, 0.2)
    specular_color: tuple = (0.0, 0.0, 0.0)
    light_location: tuple = (0.0, 0.0, 10.0)

    # 相机参数
    camera_eye: tuple = (0, 0, 10)
    camera_at: tuple = (0, 0, 0)
    camera_up: tuple = (0, 1, 0)
    camera_znear: float = 0.01
    camera_zfar: float = 100