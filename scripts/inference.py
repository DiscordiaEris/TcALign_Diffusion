import argparse
import os
import yaml
import torch
from PIL import Image
from torchvision import transforms as tv_transforms
from configs.stage2_config import Stage2Config
from models.tcalign_diffusion import TcAlignDiffusion
from data.transforms import create_transforms


def _load_config_from_yaml(yaml_path, config_cls):
    """从 YAML 文件加载配置并覆盖默认值"""
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    valid_keys = {f.name for f in config_cls.__dataclass_fields__.values()}
    filtered = {k: v for k, v in data.items() if k in valid_keys}
    return config_cls(**filtered)


def main(argv=None):
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Inference with TcALign Diffusion")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--text", type=str, required=True, help="Target text description")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default="./output.png", help="Output path")
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--clip_local_path", type=str, help="Local path to CLIP model folder")
    parser.add_argument("--diffusion_local_path", type=str, help="Local path to Stable Diffusion model folder")
    args = parser.parse_args(argv)

    # 加载配置: YAML 文件或默认值
    if args.config:
        if not os.path.isfile(args.config):
            raise FileNotFoundError(f"Config file not found: {args.config}")
        config = _load_config_from_yaml(args.config, Stage2Config)
    else:
        config = Stage2Config()

    # CLI 参数覆盖
    if args.clip_local_path:
        config.clip_local_path = args.clip_local_path
    if args.diffusion_local_path:
        config.diffusion_local_path = args.diffusion_local_path

    # 加载检查点
    model = TcAlignDiffusion(config)

    # 加载检查点
    checkpoint = torch.load(args.checkpoint, map_location=config.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 加载和预处理图像
    transform = create_transforms(config.image_size, split="val")
    image = Image.open(args.image).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(config.device)

    # 生成图像
    with torch.no_grad():
        # 获取源图像的3D参数 (theta_0: [1, total_3d_dim])
        theta_0 = model.get_3d_params(image_tensor)

        # 获取源图像的渲染图
        rendered_0 = model.render_from_params(theta_0)

        # 提取CLIP嵌入
        e_img = model.clip_encoder.encode_image(rendered_0)
        e_txt = model.clip_encoder.encode_text([args.text])

        # 通过3D编辑器预测3D参数变化 (delta_theta: [1, total_3d_dim])
        delta_theta = model.text_3d_editor(e_img, e_txt, [args.text])

        # 编码源图像到潜在空间
        source_latents = model.encode_source_images(image_tensor)

        # 通过扩散映射网络获取条件嵌入
        condition_embeddings = model.diffusion_mapper(delta_theta, source_latents)

        # 使用扩散模型生成图像
        generated_latents = model.diffusion_unet.sample(
            condition_embeddings,
            source_images=image_tensor,
            source_latents=source_latents,
            guidance_scale=config.guidance_scale,
            num_inference_steps=config.num_inference_steps
        )

        # 解码潜在代码
        generated_image = model.diffusion_unet.decode_latent(generated_latents)

    # 保存结果
    generated_image = generated_image.squeeze(0).cpu()
    generated_image = (generated_image + 1) / 2  # 反归一化 [-1,1] -> [0,1]
    generated_image = torch.clamp(generated_image, 0, 1)

    to_pil = tv_transforms.ToPILImage()
    pil_image = to_pil(generated_image)
    pil_image.save(args.output)

    print(f"Generated image saved to {args.output}")


if __name__ == "__main__":
    main()