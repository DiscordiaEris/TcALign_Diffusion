import torch
from torch import nn


class ConditionalDiffusionMappingNetwork(nn.Module):
    """Map editable 3D coefficients and spatial source features to SD token residuals."""

    def __init__(self, config):
        super().__init__()
        width = 256
        self.source_encoder = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, width, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(16, width),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.spatial_position = nn.Parameter(torch.empty(1, 16, width))
        self.attribute_embeddings = nn.Parameter(torch.empty(1, 3, width))
        self.delta_encoders = nn.ModuleList(
            nn.Sequential(nn.Linear(dim, width), nn.SiLU(), nn.Linear(width, width))
            for dim in (config.expression_dim, config.pose_dim, config.lighting_dim)
        )
        expression_start = config.shape_dim
        pose_start = expression_start + config.expression_dim
        self.delta_slices = (
            slice(expression_start, pose_start),
            slice(pose_start, pose_start + config.pose_dim),
            slice(-config.lighting_dim, None),
        )
        self.attribute_attention = nn.MultiheadAttention(width, 8, batch_first=True)
        self.attribute_norm = nn.LayerNorm(width)
        self.text_query = nn.Sequential(
            nn.LayerNorm(config.condition_embed_dim),
            nn.Linear(config.condition_embed_dim, width),
        )
        self.token_attention = nn.MultiheadAttention(width, 8, batch_first=True)
        self.output_proj = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, config.condition_embed_dim, bias=False),
        )
        nn.init.normal_(self.spatial_position, std=0.02)
        nn.init.normal_(self.attribute_embeddings, std=0.02)
        nn.init.normal_(self.output_proj[-1].weight, std=1e-3)

    def forward(self, delta_theta, source_latent, text_embeddings):
        spatial = self.source_encoder(source_latent).flatten(2).transpose(1, 2)
        spatial = spatial + self.spatial_position
        deltas = [delta_theta[:, group] for group in self.delta_slices]
        # Use each attribute's actual displacement to gate its attention token.
        magnitude = torch.stack([delta.abs().amax(dim=-1) for delta in deltas], dim=1)
        gates = magnitude.unsqueeze(-1).tanh()
        attributes = torch.stack(
            [encoder(delta) for encoder, delta in zip(self.delta_encoders, deltas)], dim=1
        )
        attributes = attributes + self.attribute_embeddings
        source_context, _ = self.attribute_attention(attributes, spatial, spatial, need_weights=False)
        edit_tokens = self.attribute_norm(attributes + source_context) * gates
        queries = self.text_query(text_embeddings)
        residual, _ = self.token_attention(queries, edit_tokens, edit_tokens, need_weights=False)
        # Learned attention biases must not edit the native text when all deltas are zero.
        edit_strength = gates.amax(dim=1, keepdim=True)
        return text_embeddings + self.output_proj(residual) * edit_strength
