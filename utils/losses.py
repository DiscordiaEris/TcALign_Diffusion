import torch
import torch.nn as nn
import torch.nn.functional as F

def cosine_similarity_loss(x, y):
    """余弦相似度损失"""
    return 1 - F.cosine_similarity(x, y, dim=-1).mean()

def identity_preservation_loss(source_embed, target_embed):
    """身份保持损失"""
    return cosine_similarity_loss(source_embed, target_embed)

def clip_consistency_loss(image_embed, text_embed):
    """CLIP一致性损失"""
    return cosine_similarity_loss(image_embed, text_embed)

def diffusion_reconstruction_loss(noise, noise_pred):
    """扩散重建损失"""
    return F.mse_loss(noise, noise_pred)

def three_d_consistency_loss(theta_pred, theta_target):
    """3D一致性损失"""
    return F.mse_loss(theta_pred, theta_target)