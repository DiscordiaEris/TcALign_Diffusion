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


def load_generation_checkpoint(config, path, *, use_default_guidance=False):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint["model_state_dict"]
    # Architecture is part of the checkpoint contract even with a caller's YAML.
    saved = checkpoint.get("inference_config", {})
    for name in ("faceid_adapter_path", "faceid_encoder_path", "faceid_scale", "use_photometric_head",
                 "isolate_faceid_attention", "identity_gain_only", "identity_refinement_rank"):
        if name in saved:
            setattr(config, name, saved[name])
    config.use_geometry_adapter = any(key.startswith("diffusion_unet.geometry_adapter.") for key in state)
    if config.use_geometry_adapter and use_default_guidance:
        config.guidance_scale = checkpoint.get("inference_guidance_scale", 2.5)
        config.use_bf16 = checkpoint.get("inference_use_bf16", False)
    if use_default_guidance:
        if "align_paired_targets" in checkpoint.get("training_config", {}):
            config.align_paired_targets = checkpoint["training_config"]["align_paired_targets"]
        for name in ("guidance_scale", "inference_strength", "num_inference_steps", "use_bf16"):
            if name in checkpoint.get("inference_config", {}):
                setattr(config, name, checkpoint["inference_config"][name])
    model = TcAlignDiffusion(config)
    model._init_diffusion_modules()
    model.load_checkpoint_state(state)
    return model.to(config.device).eval(), checkpoint["global_step"]


def main(argv=None):
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Inference with TcALign Diffusion")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--text", type=str, required=True, help="Target text description")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default="./output.png", help="Output path")
    parser.add_argument("--guidance", type=float, default=None, help="CFG guidance scale (override config)")
    parser.add_argument("--steps", type=int, default=None, help="DDIM inference steps (override config)")
    parser.add_argument("--strength", type=float, default=None, help="Override saved img2img noise strength (0, 1]")
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

    model, _ = load_generation_checkpoint(config, args.checkpoint, use_default_guidance=not args.config)
    guidance_scale = args.guidance if args.guidance is not None else config.guidance_scale
    num_inference_steps = args.steps if args.steps is not None else config.num_inference_steps
    strength = args.strength if args.strength is not None else config.inference_strength

    # 加载和预处理图像
    transform = create_transforms(config.image_size, split="val")
    image = Image.open(args.image).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(config.device)

    generated_image, _ = model.generate(
        image_tensor, [args.text], guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps, strength=strength, estimate_final_theta=False,
    )

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
