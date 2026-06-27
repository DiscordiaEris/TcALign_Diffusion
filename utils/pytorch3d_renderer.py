import torch
import numpy as np
from PIL import Image
import sys
import os
import math
import scipy.io as sio
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesVertex,
    TexturesUV,
    blending,
    materials
)
from pytorch3d.transforms import euler_angles_to_matrix


class PyTorch3DRenderer:
    """使用 PyTorch3D 进行 3D 人脸渲染"""

    def __init__(self, config):
        self.config = config
        self.device = config.device

        # 加载 BFM 模型
        self.bfm_model = self.load_bfm_model(config.bfm_model_path)

        # 初始化渲染器
        self.renderer = self.setup_renderer()

    def load_bfm_model(self, bfm_path):
        """加载 BFM 模型 (兼容 BFM09 原始格式和 transferBFM09 转换后的格式)"""
        bfm = sio.loadmat(bfm_path)

        # 兼容两种 BFM 格式:
        # - BFM09 原始: shapeMU, shapePC, shapeEV, expMU, expPC, expEV, texMU, texPC, texEV
        # - transferBFM09 输出: meanshape, idBase, exBase, meantex, texBase
        model = {}

        # 形状均值
        if 'meanshape' in bfm:
            model['shape_mean'] = torch.tensor(bfm['meanshape'].astype(np.float32)).squeeze().to(self.device)
        elif 'shapeMU' in bfm:
            model['shape_mean'] = torch.tensor(bfm['shapeMU'].astype(np.float32)).squeeze().to(self.device)
        else:
            raise KeyError("BFM file missing shape mean (meanshape or shapeMU)")

        # 形状基
        if 'idBase' in bfm:
            model['shape_basis'] = torch.tensor(bfm['idBase'].astype(np.float32)).to(self.device)
        elif 'shapePC' in bfm:
            model['shape_basis'] = torch.tensor(bfm['shapePC'].astype(np.float32)).to(self.device)
        else:
            raise KeyError("BFM file missing shape basis (idBase or shapePC)")

        # 表情基 (expression mean is zero by definition)
        if 'exBase' in bfm:
            model['exp_basis'] = torch.tensor(bfm['exBase'].astype(np.float32)).to(self.device)
        elif 'expPC' in bfm:
            model['exp_basis'] = torch.tensor(bfm['expPC'].astype(np.float32)).to(self.device)
        else:
            model['exp_basis'] = torch.zeros(model['shape_basis'].shape[0], 64).to(self.device)

        # 纹理均值
        if 'meantex' in bfm:
            model['tex_mean'] = torch.tensor(bfm['meantex'].astype(np.float32)).squeeze().to(self.device)
        elif 'texMU' in bfm:
            model['tex_mean'] = torch.tensor(bfm['texMU'].astype(np.float32)).squeeze().to(self.device)
        else:
            model['tex_mean'] = torch.zeros_like(model['shape_mean']).to(self.device)

        # 纹理基
        if 'texBase' in bfm:
            model['tex_basis'] = torch.tensor(bfm['texBase'].astype(np.float32)).to(self.device)
        elif 'texPC' in bfm:
            model['tex_basis'] = torch.tensor(bfm['texPC'].astype(np.float32)).to(self.device)
        else:
            model['tex_basis'] = torch.zeros_like(model['shape_basis']).to(self.device)

        # 三角面片索引 (统一转换为 0-based)
        tri = bfm.get('tri', bfm.get('tri', None))
        if tri is None:
            raise KeyError("BFM file missing triangle indices (tri)")
        model['tri'] = torch.tensor(tri.astype(np.int64)).to(self.device) - 1

        print(f"BFM loaded: {model['shape_mean'].shape[0]//3} vertices, {model['tri'].shape[0]} faces")
        return model

    def setup_renderer(self):
        """设置 PyTorch3D 渲染器"""
        # 相机参数
        R, T = look_at_view_transform(eye=((0, 0, 10),), at=((0, 0, 0),), up=((0, 1, 0),))
        cameras = FoVPerspectiveCameras(
            device=self.device,
            R=R,
            T=T,
            znear=0.01,
            zfar=100,
            fov=12.5936  # 根据 BFM 模型调整
        )

        # 光照设置
        lights = PointLights(
            device=self.device,
            ambient_color=((0.8, 0.8, 0.8),),
            diffuse_color=((0.2, 0.2, 0.2),),
            specular_color=((0.0, 0.0, 0.0),),
            location=((0.0, 0.0, 10.0),)
        )

        # 光栅化设置
        raster_settings = RasterizationSettings(
            image_size=self.config.render_image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size=0,
        )

        # 混合设置
        blend_params = blending.BlendParams(
            background_color=(0, 0, 0),
            sigma=1e-4,
            gamma=1e-4
        )

        # 创建渲染器
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings
            ),
            shader=SoftPhongShader(
                device=self.device,
                cameras=cameras,
                lights=lights,
                blend_params=blend_params
            )
        )

        return renderer

    def params_to_vertices(self, params):
        """将3D参数转换为顶点 [batch_size, num_vertices, 3]"""
        shape_coeff = params['shape'][:, :self.config.shape_dim]
        exp_coeff = params['expression'][:, :self.config.expression_dim]
        batch_size = shape_coeff.shape[0]

        # shape_basis: [N*3, dim], shape_coeff.T: [dim, batch] → [N*3, batch]
        shape = self.bfm_model['shape_mean'].unsqueeze(1) + \
                self.bfm_model['shape_basis'][:, :self.config.shape_dim] @ shape_coeff.T
        shape = shape + \
                self.bfm_model['exp_basis'][:, :self.config.expression_dim] @ exp_coeff.T

        # 转置并重塑: [N*3, batch] → [batch, N, 3]
        vertices = shape.T.reshape(batch_size, -1, 3)

        return vertices

    def params_to_textures(self, params):
        """将3D参数转换为纹理 [batch_size, num_vertices, 3]"""
        tex_coeff = params['texture'][:, :self.config.texture_dim]
        batch_size = tex_coeff.shape[0]

        # tex_basis: [N*3, dim], tex_coeff.T: [dim, batch] → [N*3, batch]
        texture = self.bfm_model['tex_mean'].unsqueeze(1) + \
                  self.bfm_model['tex_basis'][:, :self.config.texture_dim] @ tex_coeff.T

        # 转置并重塑: [N*3, batch] → [batch, N, 3]
        texture = texture.T.reshape(batch_size, -1, 3)

        # 归一化到 [0, 1] 范围 (逐样本 min-max，保留完整对比度)
        texture_min = texture.amin(dim=(1, 2), keepdim=True)
        texture_max = texture.amax(dim=(1, 2), keepdim=True)
        texture = (texture - texture_min) / (texture_max - texture_min + 1e-8)

        return texture

    def apply_pose(self, vertices, pose_params):
        """应用姿态变换"""
        # 提取旋转角度
        angles = pose_params[:, :3]  # [pitch, yaw, roll]

        # 创建旋转矩阵
        rotation_matrix = euler_angles_to_matrix(angles, "XYZ")

        # 应用旋转
        vertices = torch.bmm(vertices, rotation_matrix)

        return vertices

    def render(self, params):
        """渲染3D人脸, 返回 [-1, 1] 范围的图像 [batch_size, 3, H, W]"""
        # 整个渲染管线必须在 float32 下运行:
        # - BFM 基矩阵 (107127×80) 在 float16 下 matmul 会溢出
        # - PyTorch3D rasterizer 完全不支持 float16
        with torch.cuda.amp.autocast(enabled=False):
            # 将参数转换为顶点
            vertices = self.params_to_vertices(params)

            # 应用姿态变换
            vertices = self.apply_pose(vertices, params['pose'])

            # 将参数转换为纹理
            textures = self.params_to_textures(params)

            # 确保 float32 (AMP 上下文中的张量可能是 float16)
            vertices = vertices.float()
            textures = textures.float()

            # 创建纹理对象
            textures = TexturesVertex(verts_features=textures)

            # 创建网格
            faces = self.bfm_model['tri'].unsqueeze(0).repeat(vertices.shape[0], 1, 1)
            mesh = Meshes(verts=vertices, faces=faces, textures=textures)

            # 渲染图像 (PyTorch3D 输出 [0, 1])
            images = self.renderer(mesh)

        # 提取RGB通道: [batch, H, W, 4] -> [batch, 3, H, W]
        images = images[..., :3].permute(0, 3, 1, 2)

        # 从 [0, 1] 转换到 [-1, 1]，与项目数据管线保持一致
        images = images * 2.0 - 1.0

        return images

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