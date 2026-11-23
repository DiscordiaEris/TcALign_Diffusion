import os

import torch
from PIL import Image
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    MeshRasterizer,
    RasterizationSettings,
    TexturesVertex,
    look_at_view_transform,
)
from pytorch3d.structures import Meshes

from Deep3DFR.models.bfm import ParametricFaceModel


class PyTorch3DRenderer:
    """Rasterize BFM geometry with Deep3DFR pose and spherical-harmonic lighting."""

    def __init__(self, config):
        self.config = config
        self.device = config.device
        folder, name = os.path.split(config.bfm_model_path)
        if not os.path.isfile(config.bfm_model_path):
            raise FileNotFoundError(config.bfm_model_path)
        self.bfm = ParametricFaceModel(
            bfm_folder=folder, default_name=name, is_train=False,
            focal=config.camera_f, center=config.camera_c[0],
        )
        self.bfm.to(self.device)
        rotation, translation = look_at_view_transform(
            eye=((0, 0, self.bfm.camera_distance),), at=((0, 0, 0),)
        )
        cameras = FoVPerspectiveCameras(
            device=self.device, R=rotation, T=translation,
            znear=0.01, zfar=100, fov=12.5936,
        )
        self.rasterizer = MeshRasterizer(
            cameras=cameras,
            raster_settings=RasterizationSettings(
                image_size=config.render_image_size, faces_per_pixel=1,
                # Fine rasterization over every pixel/face is prohibitively slow.
                max_faces_per_bin=self.bfm.face_buf.shape[0],
            ),
        )

    def render(self, params, *, geometry=False):
        with torch.cuda.amp.autocast(enabled=False):
            params = {name: value.float() for name, value in params.items()}
            vertices = self.bfm.compute_shape(params["shape"], params["expression"])
            rotation = self.bfm.compute_rotation(params["pose"])
            normals = self.bfm.compute_norm(vertices) @ rotation
            texture = self.bfm.compute_texture(params["texture"])
            colors = self.bfm.compute_color(texture, normals, params["lighting"])
            features = torch.cat((colors, normals), dim=-1) if geometry else colors
            meshes = Meshes(
                verts=vertices @ rotation,
                faces=self.bfm.face_buf.expand(vertices.shape[0], -1, -1),
                textures=TexturesVertex(verts_features=features),
            )
            fragments = self.rasterizer(meshes)
            visible = fragments.pix_to_face >= 0
            sampled = meshes.sample_textures(fragments)[..., 0, :].masked_fill(~visible, 0)
            rgb = sampled[..., :3].clamp(0, 1).mul(2).sub(1)
            if geometry:
                normals = torch.nn.functional.normalize(sampled[..., 3:], dim=-1)
                rgb = torch.cat((rgb, normals, visible.to(rgb.dtype)), dim=-1)
            return rgb.permute(0, 3, 1, 2)

    def render_from_params_dict(self, params_dict):
        """从参数字典渲染图像"""
        # 将参数字典转换为张量格式
        params = {
            'shape': torch.tensor(params_dict['shape']).to(self.device),
            'expression': torch.tensor(params_dict['expression']).to(self.device),
            'pose': torch.tensor(params_dict['pose']).to(self.device),
            'texture': torch.tensor(params_dict['texture']).to(self.device),
            'lighting': torch.tensor(params_dict['lighting']).to(self.device),
        }

        # 渲染图像
        images = self.render(params)

        return images

    def tensor_to_pil(self, tensor):
        """将张量转换为PIL图像 (输入范围: [-1, 1])"""
        # 从张量中提取图像 [C, H, W]
        image = tensor.squeeze(0).cpu()

        # 从 [-1, 1] 转换到 [0, 255]
        image = (image + 1.0) / 2.0 * 255.0
        image = torch.clamp(image, 0, 255).byte()

        # 转换为PIL图像 [C, H, W] -> [H, W, C]
        image = image.permute(1, 2, 0).numpy()
        pil_image = Image.fromarray(image)

        return pil_image
