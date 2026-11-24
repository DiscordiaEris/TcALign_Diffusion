"""Frozen SD1.5 IP-Adapter-FaceID with its matching buffalo_l embeddings.

Weights/contracts: https://huggingface.co/h94/IP-Adapter-FaceID
The checkpoint includes both the identity attention and its companion LoRA.
Keep LoRA separate from trained base weights so loading/resuming never fuses it twice.
"""

import hashlib
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class FrozenLoRA(nn.Module):
    def __init__(self, down, up):
        super().__init__()
        self.register_buffer("down", down.detach().clone())
        self.register_buffer("up", up.detach().clone())

    def forward(self, inputs):
        return F.linear(F.linear(inputs, self.down), self.up)


def install_frozen_lora(layer, down, up):
    if hasattr(layer, "faceid_lora"):
        raise ValueError("FaceID LoRA is already installed")
    layer.add_module("faceid_lora", FrozenLoRA(down.to(layer.weight), up.to(layer.weight)))
    layer.register_forward_hook(lambda module, args, output: output + module.faceid_lora(args[0]))


def install_faceid_adapter(unet, checkpoint_path, scale):
    weights = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    for index, name in enumerate(list(unet.attn_processors)):
        attention = unet.get_submodule(name.removesuffix(".processor"))
        for tag, target in (("to_q", "to_q"), ("to_k", "to_k"),
                            ("to_v", "to_v"), ("to_out", "to_out.0")):
            prefix = f"{index}.{tag}_lora."
            install_frozen_lora(attention.get_submodule(target),
                                weights["ip_adapter"][prefix + "down.weight"],
                                weights["ip_adapter"][prefix + "up.weight"])
    # Diffusers' built-in conversion preserves pretrained text attention and
    # adds a separately normalized identity K/V branch and the trained projection.
    unet._load_ip_adapter_weights(weights)
    unet.encoder_hid_proj.requires_grad_(False)
    for processor in unet.attn_processors.values():
        if hasattr(processor, "scale"):
            processor.scale = [scale]
            processor.requires_grad_(False)


class FaceIDSourceEncoder:
    """InsightFace buffalo_l ArcFace via OpenCV DNN; never substitute FaceNet/SFace."""

    def __init__(self, weights_path, processor, cache_size=2048):
        if not Path(weights_path).is_file():
            raise FileNotFoundError(weights_path)
        self.net = cv2.dnn.readNetFromONNX(str(weights_path))
        self.processor = processor
        self.cache_size = cache_size
        self.cache = OrderedDict()
        self.landmarks = np.array([
            [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
            [41.5493, 92.3655], [70.7299, 92.2041],
        ], dtype=np.float32)

    def encode(self, images):
        embeddings = []
        for image in images.detach():
            rgb = ((image.float().clamp(-1, 1) + 1) * 127.5).round().byte()
            rgb = rgb.permute(1, 2, 0).cpu().numpy()
            key = hashlib.sha256(rgb.tobytes()).digest()
            if key not in self.cache:
                face = self.processor.detect_yunet_face(rgb)
                if face is None:
                    raise ValueError("Source face not detected for the FaceID identity condition")
                points = face[4:14].reshape(5, 2).astype(np.float32)
                affine, _ = cv2.estimateAffinePartial2D(points, self.landmarks, method=cv2.LMEDS)
                crop = cv2.warpAffine(rgb, affine, (112, 112))
                # rgb is already RGB, unlike InsightFace's usual BGR image input.
                self.net.setInput(cv2.dnn.blobFromImage(
                    crop, 1 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=False))
                embedding = self.net.forward().reshape(512).copy()
                norm = np.linalg.norm(embedding)
                if not np.isfinite(embedding).all() or norm < 1e-8:
                    raise ValueError("Invalid InsightFace source embedding")
                self.cache[key] = embedding / norm
                if len(self.cache) > self.cache_size:
                    self.cache.popitem(last=False)
            self.cache.move_to_end(key)
            embeddings.append(self.cache[key])
        return torch.from_numpy(np.stack(embeddings)).to(images.device)[:, None, :]
