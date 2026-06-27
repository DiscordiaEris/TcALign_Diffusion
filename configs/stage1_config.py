from dataclasses import dataclass
from .base_config import BaseConfig
from typing import Optional

@dataclass
class Stage1Config(BaseConfig):
    # 训练参数
    num_epochs: int = 50
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.999

    # 模型参数
    clip_embed_dim: int = 768  # CLIP ViT-L/14 joint embedding space dimension
    aes_num_attributes: int = 5  # 属性数量

    # 日志和保存
    log_interval: int = 100
    save_interval: int = 2000
    eval_interval: int = 1000000  # 实际每 epoch 结束时 eval，这里设为极大值禁用中途 eval

    # 恢复训练
    resume_from_checkpoint: Optional[str] = None