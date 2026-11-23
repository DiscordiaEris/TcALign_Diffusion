from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class FaceIdentityLoss(nn.Module):
    """Frozen VGGFace2 FaceNet with gradients through aligned generated crops."""

    def __init__(self, weights_path):
        super().__init__()
        if not weights_path or not Path(weights_path).is_file():
            raise FileNotFoundError(f"FaceNet VGGFace2 weights not found: {weights_path}")
        from facenet_pytorch import InceptionResnetV1

        encoder = InceptionResnetV1(pretrained=None)
        weights = torch.load(weights_path, map_location="cpu", weights_only=True)
        # The upstream checkpoint includes a classification head that is unused for embeddings.
        weights = {key: value for key, value in weights.items() if not key.startswith("logits.")}
        encoder.load_state_dict(weights, strict=True)
        self.encoder = encoder.requires_grad_(False).eval()

    def train(self, mode=True):
        return super().train(False)

    def embed(self, images, alignment):
        grid = F.affine_grid(alignment.float(), (len(images), 3, 160, 160), align_corners=False)
        crops = F.grid_sample(images.float(), grid, mode="bilinear",
                              padding_mode="border", align_corners=False)
        return self.encoder(crops.clamp(-1, 1) * (127.5 / 128.0))

    def forward(self, generated, source, target, source_alignment, target_alignment):
        with torch.amp.autocast(device_type=generated.device.type, enabled=False):
            with torch.no_grad():
                source_embedding = self.embed(source, source_alignment)
                target_embedding = self.embed(target, target_alignment)
            embedding = self.embed(generated, target_alignment)
            source_distance = 1 - F.cosine_similarity(embedding, source_embedding)
            target_distance = 1 - F.cosine_similarity(embedding, target_embedding)
            return (0.65 * source_distance + 0.35 * target_distance).mean()
