import torch
from torch import nn
from torch.nn import functional as F


def align_geometry(geometry, alignment, output_size):
    """Warp canonical face planes into the source photo's frame."""
    with torch.amp.autocast(device_type=geometry.device.type, enabled=False):
        bottom = alignment.new_tensor([0, 0, 1]).expand(len(alignment), 1, 3)
        inverse = torch.linalg.inv(torch.cat((alignment, bottom), dim=1).float())[:, :2]
        grid = F.affine_grid(inverse, (len(geometry), 7, *output_size), align_corners=False)
        planes = torch.cat((geometry[:, :3].float() + 1, geometry[:, 3:].float()), dim=1)
        warped = F.grid_sample(planes, grid, mode="bilinear", align_corners=False)
        return torch.cat((warped[:, :3] - 1, warped[:, 3:]), dim=1)


class GeometryAdapter(nn.Module):
    """Convert rasterized RGB, normals and visibility into UNet spatial residuals."""

    def __init__(self, channels):
        super().__init__()
        widths = [64 * min(2 ** stage, 4) for stage in range(len(channels))]
        self.blocks = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(7 if index == 0 else widths[index - 1], width, 3,
                          stride=1 if index == 0 else 2, padding=1),
                nn.GroupNorm(8, width),
                nn.SiLU(),
                nn.Conv2d(width, width, 3, padding=1),
                nn.GroupNorm(8, width),
                nn.SiLU(),
            )
            for index, width in enumerate(widths)
        )
        self.projections = nn.ModuleList(
            nn.Conv2d(width, channels_out, 1) for width, channels_out in zip(widths, channels)
        )
        for projection in self.projections:
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)

    def forward(self, geometry, latent_size):
        features = F.interpolate(geometry, size=latent_size, mode="bilinear",
                                 align_corners=False, antialias=True)
        outputs = []
        for block, projection in zip(self.blocks, self.projections):
            features = block(features)
            outputs.append(projection(features))
        return outputs
