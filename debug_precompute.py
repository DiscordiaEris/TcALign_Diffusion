#!/usr/bin/env python3
"""诊断单张图片的 precompute 流程"""
import sys, os, traceback
import torch, numpy as np
from PIL import Image

from configs.stage1_config import Stage1Config
from utils.deep3d_utils import load_deep3d_model
from utils.pytorch3d_renderer import PyTorch3DRenderer
from data.transforms import create_transforms

config = Stage1Config()
config.data_root = "/workspace/HRTao/git/TcALign_Diffusion/datasets/ffhq"

# 找第一张图测试
image_dir = os.path.join(config.data_root, "images", "train")
images = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
if not images:
    print("ERROR: No images found!")
    sys.exit(1)

test_img = images[0]
img_path = os.path.join(image_dir, test_img)
print(f"Testing: {test_img}")

try:
    # 1. 加载模型
    print("1. Loading Deep3D processor...")
    processor = load_deep3d_model(config)
    print("   OK")

    print("2. Loading PyTorch3D renderer...")
    renderer = PyTorch3DRenderer(config)
    print("   OK")

    # 2. 预处理图像
    print(f"3. Opening image...")
    image = Image.open(img_path).convert('RGB')
    print(f"   Size: {image.size}")

    transform = create_transforms(config.image_size, split="train")
    image_tensor = transform(image).unsqueeze(0)
    print(f"   Tensor shape: {image_tensor.shape}, range: [{image_tensor.min():.2f}, {image_tensor.max():.2f}]")

    # 3. 3D 参数提取
    print("4. Extracting 3D params...")
    params = processor.process(image_tensor)
    for k, v in params.items():
        print(f"   {k}: shape={v.shape}")
    print("   OK")

    # 4. 渲染
    print("5. Rendering...")
    rendered = renderer.render_from_params_dict(params)
    print(f"   Rendered shape: {rendered.shape}")
    print("   OK")

    print("\n✅ All steps passed!")
except Exception as e:
    print(f"\n❌ Failed at step:")
    traceback.print_exc()
