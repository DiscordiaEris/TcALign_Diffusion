import torch
import torchvision.transforms as T
from torch.nn import functional as F


def align_paired_target(target_image, source_alignment, target_alignment):
    """Register a paired target portrait into the source portrait's face frame.

    Cached alignments map canonical face coordinates to photo coordinates.
    Sampling the registered image with the source alignment therefore recovers
    the original target sampled with its target alignment.
    """
    bottom = source_alignment.new_tensor([[0, 0, 1]])
    source = torch.cat((source_alignment, bottom), dim=0).float()
    target = torch.cat((target_alignment, bottom), dim=0).float()
    sampling_transform = (target @ torch.linalg.inv(source))[:2]
    grid = F.affine_grid(sampling_transform[None], (1, *target_image.shape),
                         align_corners=False)
    return F.grid_sample(target_image[None], grid, mode="bilinear",
                         padding_mode="border", align_corners=False)[0]


def create_transforms(image_size, split="train"):
    """创建图像变换"""
    if split == "train":
        transform = T.Compose([
            T.Resize(image_size),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomAffine(degrees=10, translate=(0.1, 0.1)),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize([0.5], [0.5])
        ])
    else:
        transform = T.Compose([
            T.Resize(image_size),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize([0.5], [0.5])
        ])

    return transform


def create_rendered_transform(image_size):
    """为渲染图创建确定性变换 (不需要随机增强, 保证 CLIP 嵌入可缓存)"""
    return T.Compose([
        T.Resize(image_size),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize([0.5], [0.5])
    ])
