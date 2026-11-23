"""
图像质量评估指标模块。

提供标准的图像质量指标计算:
  - SSIM  (结构相似性): 衡量图像结构的相似程度, 值越高越好
  - PSNR  (峰值信噪比): 衡量像素级保真度, 值越高越好
  - LPIPS (感知相似度): 衡量感知层面的相似性, 值越低越好 (需要 pip install lpips)
  - FID   (Fréchet Inception Distance): 衡量生成图像分布的真实性, 值越低越好

所有公开方法接受 [-1, 1] 范围的图像张量 (项目约定), 内部按需转换为 [0, 1]。
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────
# SSIM 实现 (纯 PyTorch, 无需 skimage, 留在 GPU 上)
# ──────────────────────────────────────────────

def _gaussian_kernel(size: int = 11, sigma: float = 1.5, channels: int = 3) -> torch.Tensor:
    """创建 2D 高斯核"""
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    g_2d = g[:, None] * g[None, :]  # [size, size]
    kernel = g_2d.expand(channels, 1, size, size).contiguous()
    return kernel


def _ssim_per_channel(
    img1: torch.Tensor, img2: torch.Tensor,
    kernel: torch.Tensor, C1: float = 0.01 ** 2, C2: float = 0.03 ** 2,
) -> torch.Tensor:
    """
    计算单张图像对的 SSIM。

    Args:
        img1, img2: [B, C, H, W], 范围 [0, 1]
        kernel: 高斯核 [C, 1, K, K]
    Returns:
        ssim: [B] 每张图像的平均 SSIM (跨通道)
    """
    # 计算局部统计量
    mu1 = F.conv2d(img1, kernel, padding=kernel.shape[-1] // 2, groups=img1.shape[1])
    mu2 = F.conv2d(img2, kernel, padding=kernel.shape[-1] // 2, groups=img2.shape[1])

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, kernel, padding=kernel.shape[-1] // 2, groups=img1.shape[1]) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, kernel, padding=kernel.shape[-1] // 2, groups=img2.shape[1]) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, kernel, padding=kernel.shape[-1] // 2, groups=img2.shape[1]) - mu1_mu2

    # SSIM 公式
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean(dim=[1, 2, 3])  # [B]


# ──────────────────────────────────────────────
# 图像质量指标主类
# ──────────────────────────────────────────────

class ImageQualityMetrics:
    """图像质量指标计算器。

    两类指标:
      - Per-sample (立即计算, 每对图像):
          SSIM, PSNR, LPIPS
      - Distribution (跨 batch 累积):
          FID (Fréchet Inception Distance)

    像素范围: 公开方法接受 [-1, 1], 内部自动转换。
    """

    def __init__(self, config, device: str = "cuda"):
        """
        Args:
            config: Stage2Config 实例 (或任何包含指标开关字段的 dataclass)
            device: 计算设备
        """
        self.device = device
        self.config = config

        # 读取开关
        self.ssim_enabled = getattr(config, 'ssim_enabled', True)
        self.psnr_enabled = getattr(config, 'psnr_enabled', True)
        self.lpips_enabled = getattr(config, 'lpips_enabled', False)
        self.fid_enabled = getattr(config, 'fid_enabled', True)
        self._fid_dims = getattr(config, 'fid_dims', 2048)

        # ── SSIM 高斯核 (惰性创建) ──
        self._ssim_kernel: torch.Tensor | None = None

        # ── LPIPS 模型 (惰性加载) ──
        self._lpips_model = None
        self._HAS_LPIPS = False
        if self.lpips_enabled:
            self._init_lpips()

        # ── FID: InceptionV3 特征提取器 ──
        self._inception_model = None
        self._real_features: list[torch.Tensor] = []
        self._gen_features: list[torch.Tensor] = []
        if self.fid_enabled:
            self._init_inception()

    # ── 初始化方法 ──────────────────────────────────

    def _init_lpips(self):
        """惰性加载 LPIPS 模型"""
        try:
            import lpips  # noqa: F811
            self._lpips_model = lpips.LPIPS(net='alex').to(self.device)
            self._lpips_model.eval()
            for p in self._lpips_model.parameters():
                p.requires_grad = False
            self._HAS_LPIPS = True
        except ImportError:
            self._HAS_LPIPS = False
            print("[ImageQualityMetrics] WARNING: lpips not installed. "
                  "LPIPS metric disabled. Install with: pip install lpips")

    def _init_inception(self):
        """惰性加载 InceptionV3 特征提取器"""
        from torchvision.models import inception_v3
        self._inception_model = inception_v3(
            pretrained=True, transform_input=False
        ).to(self.device)
        self._inception_model.eval()
        # 移除辅助分类器, 使用 avgpool 输出作为特征 (2048-d)
        self._inception_model.fc = nn.Identity()
        # 设置 aux_logits=False 避免推理时报错
        self._inception_model.aux_logits = False
        for p in self._inception_model.parameters():
            p.requires_grad = False

    # ── 像素范围转换 ────────────────────────────────

    @staticmethod
    def _to_01(images: torch.Tensor) -> torch.Tensor:
        """将 [-1, 1] 转换为 [0, 1]"""
        return (images + 1.0) / 2.0

    @staticmethod
    def _to_uint8(images: torch.Tensor) -> torch.Tensor:
        """将 [-1, 1] 转换为 [0, 255] uint8"""
        return ((images + 1.0) / 2.0 * 255.0).clamp(0, 255).to(torch.uint8)

    # ── SSIM ───────────────────────────────────────

    @torch.no_grad()
    def compute_ssim(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """
        计算 batch SSIM。

        Args:
            img1, img2: [B, C, H, W], 范围 [-1, 1]
        Returns:
            平均 SSIM (标量 float)
        """
        if self._ssim_kernel is None or self._ssim_kernel.device != img1.device:
            self._ssim_kernel = _gaussian_kernel(channels=img1.shape[1]).to(img1.device)

        # 转换为 [0, 1]
        img1_01 = self._to_01(img1)
        img2_01 = self._to_01(img2)

        return _ssim_per_channel(img1_01, img2_01, self._ssim_kernel).mean().item()

    # ── PSNR ───────────────────────────────────────

    @torch.no_grad()
    def compute_psnr(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """
        计算 batch PSNR。

        Args:
            img1, img2: [B, C, H, W], 范围 [-1, 1]
        Returns:
            平均 PSNR (dB, 标量 float)
        """
        # 转换为 [0, 1]
        img1_01 = self._to_01(img1)
        img2_01 = self._to_01(img2)

        mse = F.mse_loss(img1_01, img2_01, reduction='none').mean(dim=[1, 2, 3])
        # 防止 log(0)
        mse = torch.clamp(mse, min=1e-10)
        psnr = 10.0 * torch.log10(1.0 / mse)
        return psnr.mean().item()

    # ── LPIPS ──────────────────────────────────────

    @torch.no_grad()
    def compute_lpips(self, img1: torch.Tensor, img2: torch.Tensor) -> float | None:
        """
        计算 batch LPIPS。

        Args:
            img1, img2: [B, C, H, W], 范围 [-1, 1]
        Returns:
            平均 LPIPS (标量 float), 如果 LPIPS 不可用返回 None
        """
        if not self._HAS_LPIPS or self._lpips_model is None:
            return None

        # LPIPS alex net 期望 [-1, 1]
        score = self._lpips_model(img1, img2)
        return score.mean().item()

    # ── 综合 per-sample 指标 ────────────────────────

    @torch.no_grad()
    def compute_per_sample_metrics(
        self, source: torch.Tensor, generated: torch.Tensor
    ) -> dict[str, float]:
        """
        计算单个 batch 的所有 per-sample 指标。

        Args:
            source: 源图像 [B, C, H, W], 范围 [-1, 1]
            generated: 生成图像 [B, C, H, W], 范围 [-1, 1]
        Returns:
            指标字典, 键如 "ssim", "psnr", "lpips"
        """
        metrics: dict[str, float] = {}

        if self.ssim_enabled:
            metrics["ssim"] = self.compute_ssim(source, generated)

        if self.psnr_enabled:
            metrics["psnr"] = self.compute_psnr(source, generated)

        if self.lpips_enabled:
            lpips_val = self.compute_lpips(source, generated)
            if lpips_val is not None:
                metrics["lpips"] = lpips_val

        return metrics

    # ── FID ────────────────────────────────────────

    @torch.no_grad()
    def extract_inception_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        提取 InceptionV3 pool3 特征。

        Args:
            images: [B, C, H, W], 范围 [-1, 1]
        Returns:
            特征张量 [B, 2048]
        """
        if self._inception_model is None:
            raise RuntimeError("InceptionV3 not initialized; set fid_enabled=True")

        # 转换到 [0, 1], 然后 resize 到 299x299
        images_01 = self._to_01(images)
        images_299 = F.interpolate(images_01, size=(299, 299), mode='bilinear', align_corners=False)

        features = self._inception_model(images_299)  # [B, 2048]
        return features

    @torch.no_grad()
    def accumulate_real_features(self, images: torch.Tensor) -> None:
        """累积真实图像 (验证集源图像) 的 InceptionV3 特征"""
        features = self.extract_inception_features(images)
        self._real_features.append(features.cpu())

    @torch.no_grad()
    def accumulate_gen_features(self, images: torch.Tensor) -> None:
        """累积生成图像 的 InceptionV3 特征"""
        features = self.extract_inception_features(images)
        self._gen_features.append(features.cpu())

    @torch.no_grad()
    def compute_fid(self) -> float:
        """
        使用累积的特征计算 FID。

        FID = ||μ₁ - μ₂||² + Tr(Σ₁ + Σ₂ - 2√(Σ₁Σ₂))

        Returns:
            FID 分数 (float)
        """
        if not self._real_features or not self._gen_features:
            raise RuntimeError(
                "No features accumulated. Call accumulate_real_features() and "
                "accumulate_gen_features() before compute_fid()."
            )

        real_feats = torch.cat(self._real_features, dim=0).numpy()  # [N, 2048]
        gen_feats = torch.cat(self._gen_features, dim=0).numpy()    # [M, 2048]

        return self._calculate_fid(real_feats, gen_feats)

    def reset_fid_state(self) -> None:
        """清空 FID 特征累积器"""
        self._real_features = []
        self._gen_features = []

    @staticmethod
    def _calculate_fid(real_feats: "np.ndarray", gen_feats: "np.ndarray") -> float:
        """
        计算两组特征之间的 Fréchet Inception Distance。

        Args:
            real_feats: [N, D] 真实图像特征
            gen_feats:  [M, D] 生成图像特征
        Returns:
            FID 分数
        """
        import numpy as np

        mu1 = np.mean(real_feats, axis=0)
        mu2 = np.mean(gen_feats, axis=0)
        sigma1 = np.cov(real_feats, rowvar=False)
        sigma2 = np.cov(gen_feats, rowvar=False)

        # 数值稳定性: 加入微小对角扰动
        eps = 1e-6
        sigma1 = sigma1 + np.eye(sigma1.shape[0]) * eps
        sigma2 = sigma2 + np.eye(sigma2.shape[0]) * eps

        # 均值差的平方和
        diff = mu1 - mu2
        mean_dist = np.dot(diff, diff)

        # 协方差矩阵乘积的平方根
        from scipy.linalg import sqrtm
        cov_mean = sqrtm(sigma1 @ sigma2)

        # 去掉可能出现的复数部分 (数值误差)
        if np.iscomplexobj(cov_mean):
            cov_mean = cov_mean.real

        fid = mean_dist + np.trace(sigma1 + sigma2 - 2 * cov_mean)
        return float(fid)
