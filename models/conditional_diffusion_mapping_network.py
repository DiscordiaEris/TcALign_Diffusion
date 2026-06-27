import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalDiffusionMappingNetwork(nn.Module):
    """条件扩散映射网络 - 接收 Δθ 和源图像编码作为输入"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # 源图像编码器 (将VAE潜在表示映射到与Δθ相同的空间)
        self.source_encoder = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化
            nn.Flatten(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 128)  # 输出与Δθ拼接前的维度
        )

        # Δθ 投影层
        self.delta_theta_projection = nn.Sequential(
            nn.Linear(config.total_3d_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128)  # 输出与源图像编码拼接前的维度
        )

        # 融合层 (拼接后的处理)
        self.fusion_network = nn.Sequential(
            nn.Linear(256, 512),  # 128 (图像) + 128 (Δθ) = 256
            nn.ReLU(),
            nn.Linear(512, 512)
        )

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=512,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)

        # 位置编码
        self.position_embedding = nn.Parameter(
            torch.randn(1, config.condition_seq_length, 512)
        )

        # 输出投影层
        self.output_projection = nn.Linear(512, config.condition_embed_dim)

    def forward(self, delta_theta, source_latent):
        """
        前向传播

        参数:
            delta_theta: 3D参数差异 [batch_size, total_3d_dim]
            source_latent: 源图像的VAE潜在表示 [batch_size, 4, H, W]

        返回:
            condition_embeddings: 条件嵌入 [batch_size, seq_len, embed_dim]
        """
        batch_size = delta_theta.shape[0]

        # 编码源图像
        source_encoded = self.source_encoder(source_latent)  # [batch_size, 128]

        # 投影Δθ
        delta_theta_encoded = self.delta_theta_projection(delta_theta)  # [batch_size, 128]

        # 融合源图像编码和Δθ编码
        fused = torch.cat([source_encoded, delta_theta_encoded], dim=-1)  # [batch_size, 256]
        fused = self.fusion_network(fused)  # [batch_size, 512]

        # 扩展到序列长度
        fused = fused.unsqueeze(1)  # [batch_size, 1, 512]
        fused = fused.expand(-1, self.config.condition_seq_length, -1)  # [batch_size, seq_len, 512]

        # 添加位置编码
        fused = fused + self.position_embedding

        # 通过Transformer编码器
        condition_embeddings = self.transformer_encoder(fused)

        # 输出投影
        condition_embeddings = self.output_projection(condition_embeddings)

        return condition_embeddings