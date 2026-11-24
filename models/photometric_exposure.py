"""Learned scalar exposure calibration, separate from diffusion and SH lighting.

Images use the repository's [-1, 1] RGB convention. Luminance here is a
weighted mean of [0, 1] RGB, rather than a claim of linear-light exposure.
No text parsing, pretrained weights, or dataset-specific exposure labels are
part of this module. Callers provide an explicit lighting-request mask.
"""

import math
import re

import torch
from torch import nn
from torch.nn import functional as F


def explicit_lighting_requests(texts, *, device=None):
    """Route explicit exposure edits; unknown text must not enable calibration."""
    anchors = {"lighting", "illumination", "exposure", "relight", "light", "lit",
               "dim", "dimmer", "dark", "darker", "darken", "brighten", "brighter"}
    requested = []
    for text in texts:
        words = set(re.findall(r"[a-z]+", text.lower()))
        unchanged = re.search(r"\b(keep|unchanged|preserve)\b.*\b(light|lighting|illumination|exposure)\b", text.lower())
        negated = re.search(r"\b(no|not|never|without|don['’]t)\b.*\b(change|alter|dim|darken|brighten|relight)\b", text.lower())
        requested.append(bool(words & anchors) and not unchanged and not negated)
    return torch.tensor(requested, dtype=torch.bool, device=device)


class PhotometricExposureHead(nn.Module):
    """Predict absolute target log luminance from frozen text and source light.

    ``forward`` accepts native SD text tokens [B, T, D] (normally [B, 77,
    768]) and source log luminance [B]. A zero-initialized residual starts
    with the source's exposure target. This does not imply that applying the
    head to a differently exposed generated image is initially an identity.
    All learned state consists of ordinary checkpointable parameters.
    """

    def __init__(self, text_embed_dim=768, hidden_dim=32):
        super().__init__()
        if text_embed_dim < 1 or hidden_dim < 1:
            raise ValueError("Exposure head dimensions must be positive")
        self.text_embed_dim = text_embed_dim
        self.residual = nn.Sequential(
            nn.Linear(text_embed_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, text_tokens, source_log_luminance):
        if (text_tokens.ndim != 3 or text_tokens.shape[-1] != self.text_embed_dim
                or text_tokens.shape[1] == 0):
            raise ValueError("Text tokens must have shape [B, T, text_embed_dim] with T > 0")
        if source_log_luminance.shape != (len(text_tokens),):
            raise ValueError("Source log luminance must have shape [B]")
        pooled = text_tokens.float().mean(dim=1)
        features = torch.cat((pooled, source_log_luminance.float()[:, None]), dim=1)
        features = features.to(dtype=self.residual[0].weight.dtype)
        residual = self.residual(features).squeeze(-1)
        return source_log_luminance.float() + residual.float()


def aligned_face_luminance(images, alignment, *, crop_size=64):
    """Return [B] face means in [0, 1] RGB using cached crop-to-image affines.

    Border sampling matches existing face losses. Alignment and luminance
    arithmetic run in float32, including when the surrounding model uses AMP.
    """
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("Images must have shape [B, 3, H, W]")
    if alignment.shape != (len(images), 2, 3):
        raise ValueError("Alignment must have shape [B, 2, 3]")
    if crop_size < 1:
        raise ValueError("Face crop size must be positive")
    with torch.amp.autocast(device_type=images.device.type, enabled=False):
        rgb = ((images.float() + 1) * .5).clamp(0, 1)
        grid = F.affine_grid(
            alignment.to(device=images.device, dtype=torch.float32),
            (len(images), 3, crop_size, crop_size), align_corners=False,
        )
        crops = F.grid_sample(
            rgb, grid, mode="bilinear", padding_mode="border", align_corners=False,
        )
        weights = crops.new_tensor([.2126, .7152, .0722])[None, :, None, None]
        return (crops * weights).sum(dim=1).mean(dim=(-2, -1))


def apply_photometric_exposure(
    head, source_images, generated_images, text_tokens, alignment, lighting_mask,
    *, enabled=True, max_gain=8.0,
):
    """Apply one differentiable RGB gain to each explicitly requested image.

    ``head`` predicts an absolute face mean using source pixels and frozen
    text tokens. The gain uses the *generated* face mean as its denominator,
    so it corrects sampler exposure drift instead of repeatedly scaling the
    source's exposure. It scales the whole image, without spatial filtering.

    Disabled calls and batches with no requested rows return the original
    tensor. Unrequested rows in mixed batches retain their exact values and
    dtype, even outside [-1, 1]. Only requested rows are clamped to that range.

    Black inputs remain finite and cannot acquire missing texture. A 1e-4
    luminance floor bounds division and log evaluation. Target luminance is
    restricted to [1e-4, 1]. Brightening is limited by ``max_gain`` and RGB
    saturation: after clipping, the requested target mean may be unreachable.
    """
    if not enabled:
        return generated_images
    if lighting_mask.dtype != torch.bool:
        raise TypeError("Lighting mask must be an explicit boolean tensor")
    if lighting_mask.shape != (len(generated_images),):
        raise ValueError("Lighting mask must have shape [B]")
    active = lighting_mask.to(device=generated_images.device).nonzero(as_tuple=True)[0]
    if not len(active):
        return generated_images
    if not math.isfinite(max_gain) or max_gain <= 0:
        raise ValueError("Maximum exposure gain must be finite and positive")
    if source_images.shape != generated_images.shape:
        raise ValueError("Source and generated images must have matching [B, 3, H, W] shapes")
    if source_images.device != generated_images.device:
        raise ValueError("Source and generated images must be on the same device")
    if alignment.shape != (len(generated_images), 2, 3):
        raise ValueError("Alignment must have shape [B, 2, 3]")
    if len(text_tokens) != len(generated_images):
        raise ValueError("Text tokens must describe every image in the batch")
    affine = alignment.to(device=generated_images.device)[active]
    epsilon = 1e-4
    with torch.amp.autocast(device_type=generated_images.device.type, enabled=False):
        source_mean = aligned_face_luminance(source_images[active], affine)
        generated_mean = aligned_face_luminance(generated_images[active], affine)
        target_log = head(text_tokens[active], source_mean.clamp_min(epsilon).log())
        target_mean = target_log.float().clamp(min=math.log(epsilon), max=0).exp()
        gain = (target_mean / generated_mean.clamp_min(epsilon)).clamp(max=max_gain)
        rgb = ((generated_images[active].float() + 1) * .5).clamp(0, 1)
        corrected = (rgb * gain[:, None, None, None]).clamp(0, 1) * 2 - 1
    result = generated_images.clone()
    result[active] = corrected.to(dtype=generated_images.dtype)
    return result
