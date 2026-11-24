import re

import torch
import torch.nn as nn
from .attribute_embedding_space import AttributeEmbeddingSpace


class TextConditional3DEditor(nn.Module):
    """文本条件3D编辑器"""

    # 各属性的语义锚点词，用于判断文本涉及了哪类属性编辑
    _ATTR_ANCHORS = {
        'expression': ['smile', 'smiling', 'angry', 'surprised', 'sad', 'neutral',
                       'happy', 'frown', 'grin', 'laugh', 'expression', 'emotion'],
        'pose': ['front', 'frontal', 'turn', 'tilt', 'pose', 'look', 'looking',
                 'facing', 'angle', 'view', 'yaw', 'pitch', 'profile', 'rotate'],
        'lighting': ['bright', 'brighter', 'brighten', 'dim', 'dimmer', 'dark', 'darker', 'darken', 'natural', 'studio', 'soft',
                     'shadow', 'lighting', 'illumination', 'light', 'lit', 'relight'],
    }

    def __init__(self, config):
        super().__init__()
        self.config = config
        expression_start = config.shape_dim
        pose_start = expression_start + config.expression_dim
        lighting_start = pose_start + config.pose_dim + config.texture_dim
        self.attribute_slices = {
            "expression": slice(expression_start, pose_start),
            "pose": slice(pose_start, pose_start + config.pose_dim),
            "lighting": slice(lighting_start, config.total_3d_dim),
        }

        # 属性嵌入空间
        self.aes = AttributeEmbeddingSpace(
            embed_dim=config.clip_embed_dim,
            attribute_dims={
                'expression': 5,
                'pose': 4,
                'lighting': 4
            }
        )

        # 3D参数预测网络 (最后一层用小权重初始化，使训练从接近零的 Δθ 开始)
        self.theta_predictor = nn.Sequential(
            nn.Linear(config.clip_embed_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, config.total_3d_dim)
        )
        # 将最后一层权重初始化为合理的较小值（不是零），让初始 Δθ ≈ 0.2-0.5
        # 这样渲染结果与源图有可见差异，CLIP 能提供有效的梯度信号
        # std=0.02 → |Δθ| ≈ 0.22，足以产生可察觉但不过分的面部变化
        nn.init.normal_(self.theta_predictor[-1].weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.theta_predictor[-1].bias)

    def parse_attributes(self, text):
        """从文本中解析要修改的属性

        优先级: 精确关键词匹配 → 锚点词匹配 → 默认全属性
        对训练数据模板 "A face with {expr} expression, {pose}, {lighting}" 覆盖良好，
        对自由文本也会通过锚点词做更鲁棒的推断。
        """
        words = set(re.findall(r"[a-z]+", text.lower()))
        attributes = []
        for attr, anchors in self._ATTR_ANCHORS.items():
            if words.intersection(anchors):
                attributes.append(attr)
        if not attributes and words.intersection({"left", "right", "side"}):
            attributes.append("pose")
        # 如果没有任何匹配，默认编辑所有可用属性
        return attributes if attributes else list(self.aes.attribute_dims.keys())

    def attribute_mask(self, attributes, reference):
        mask = torch.zeros_like(reference, dtype=torch.bool)
        for row, selected in enumerate(attributes):
            for attribute in selected:
                mask[row, self.attribute_slices[attribute]] = True
        return mask

    def forward(self, e_img, e_txt, texts=None, return_manipulated=False):
        """
        前向传播

        参数:
            e_img: 源图的CLIP嵌入 [batch_size, embed_dim]
            e_txt: 目标文本的CLIP嵌入 [batch_size, embed_dim]
            texts: 目标文本列表，用于解析要修改的属性
            return_manipulated: 是否额外返回操控后的嵌入

        返回:
            delta_theta: 3D参数的变化量 [batch_size, total_3d_dim]
            (e_manipulated if return_manipulated)
        """
        batch_size = e_img.shape[0]

        # 从文本中解析要修改的属性 (每样本一个属性列表)
        if texts is None:
            target_attributes = [list(self.aes.attribute_dims.keys()) for _ in range(batch_size)]
        else:
            target_attributes = [self.parse_attributes(text) for text in texts]

        # 通过AES进行多属性操控 (支持每样本编辑多个属性)
        e_manipulated = self.aes.manipulate(e_img, e_txt, target_attributes)

        # 预测3D参数变化量（作为Δθ）
        delta_theta = self.theta_predictor(e_manipulated).float()

        # 安全裁剪：防止极端 Δθ 导致 PyTorch3D 渲染器产生 NaN
        delta_theta = torch.clamp(
            delta_theta,
            min=-self.config.delta_theta_clamp,
            max=self.config.delta_theta_clamp,
        )
        delta_theta = delta_theta.masked_fill(~self.attribute_mask(target_attributes, delta_theta), 0)

        if return_manipulated:
            return delta_theta, e_manipulated
        return delta_theta
