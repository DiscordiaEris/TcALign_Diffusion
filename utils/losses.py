import torch
import torch.nn as nn
import torch.nn.functional as F


def make_edit_training_pair(scheduler, target, source, noise, timesteps, source_anchor):
    """Mix standard diffusion with supervised source-to-target transport.

    The source branch deliberately starts with a noised *source*. Its label
    must therefore include the requested latent displacement; using the added
    Gaussian noise as the label would explicitly train copying the source.
    """
    target, source, noise = target.float(), source.float(), noise.float()
    anchor = torch.where(source_anchor[:, None, None, None], source, target)
    noisy = scheduler.add_noise(anchor, noise, timesteps)
    alpha = scheduler.alphas_cumprod.to(target.device)[timesteps, None, None, None]
    effective_noise = noise + (alpha / (1 - alpha).clamp_min(1e-8)).sqrt() * (anchor - target)
    return noisy, effective_noise

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


def requested_attribute_control_loss(predicted, source, target, attributes, groups):
    """Measure requested edit error relative to copying the source coefficients.

    Each requested attribute receives equal weight within each sample. Its
    coefficient MSE is divided by the measured source-to-target displacement
    MSE, with RMS floors for tiny edits. Copying a meaningful requested edit has
    loss one; matching it has loss zero. Unrequested target changes contribute
    no loss or gradient. Teachers and normalization stay detached.
    """
    if predicted.ndim != 2 or source.shape != predicted.shape or target.shape != predicted.shape:
        raise ValueError("Control coefficients must have matching [batch, coefficients] shapes")
    if len(attributes) != len(predicted) or not len(predicted):
        raise ValueError("Control attributes must describe every nonempty batch sample")
    floors = {"expression": 0.05, "pose": 0.1, "lighting": 0.02}
    predicted = predicted.float()
    source = source.detach().float()
    target = target.detach().float()
    samples = []
    for index, requested in enumerate(attributes):
        if not requested:
            raise ValueError("Control supervision requires at least one requested attribute")
        errors = []
        for name in requested:
            group = groups[name]
            displacement = target[index, group] - source[index, group]
            scale_squared = displacement.square().mean().clamp_min(floors[name] ** 2)
            error = predicted[index, group] - target[index, group]
            errors.append(error.square().mean() / scale_squared)
        samples.append(torch.stack(errors).mean())
    return torch.stack(samples).mean()


def requested_lighting_exposure_loss(predicted_images, source_images, target_images,
                                     alignment, attributes):
    """Supervise requested exposure directly on a shared aligned face crop.

    Luminance is measured in normalized RGB's [-1, 1] range. A meaningful
    source-copy has loss one, while target exposure has loss zero. The 0.05
    exposure floor bounds gradients when paired brightness barely changes.
    """
    indices = [index for index, requested in enumerate(attributes) if "lighting" in requested]
    if not indices:
        return predicted_images.float().sum() * 0
    with torch.amp.autocast(device_type=predicted_images.device.type, enabled=False):
        predicted = predicted_images[indices].float()
        affine = alignment[indices].detach().float()
        grid = F.affine_grid(affine, (len(indices), 3, 64, 64), align_corners=False)
        weights = predicted.new_tensor([0.2126, 0.7152, 0.0722])[None, :, None, None]

        def exposure(images):
            crop = F.grid_sample(images.float(), grid, mode="bilinear",
                                 padding_mode="border", align_corners=False)
            return (crop * weights).sum(dim=1).mean(dim=(-2, -1))

        with torch.no_grad():
            source_exposure = exposure(source_images[indices].detach())
            target_exposure = exposure(target_images[indices].detach())
            scale_squared = (target_exposure - source_exposure).square().clamp_min(0.05 ** 2)
        predicted_exposure = exposure(predicted)
        return ((predicted_exposure - target_exposure).square() / scale_squared).mean()
