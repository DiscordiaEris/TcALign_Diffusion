# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

TcALign with Diffusion — a two-stage text-conditioned face image editing framework. It uses 3D Morphable Models (3DMM) to bridge text descriptions and diffusion-based image generation.

**Stage 1**: Train a Text-Conditional 3D Editor (Γ network) that maps text → CLIP embedding manipulation in an Attribute Embedding Space (AES) → 3DMM parameter changes (Δθ).

**Stage 2**: Train a Conditional Diffusion Mapping Network (Φ_diff) that fuses Δθ with source image VAE latents to produce cross-attention conditions for a modified Stable Diffusion UNet. The UNet is modified to accept multimodal input (noisy latents + source latents + source RGB, concatenated channel-wise).

## Common Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Verify setup (checks Deep3DFR repo, BFM model, weights, Python packages)
python verify_setup.py

# Train Stage 1 (3D Editor)
python main.py train_stage1

# Train Stage 2 (Diffusion Mapping) — requires Stage 1 checkpoint
python main.py train_stage2 --stage1_ckpt checkpoints/final_checkpoint_epoch_49.pth

# Resume training from a checkpoint
python main.py train_stage1 --resume checkpoints/checkpoint_step_5000.pth

# Inference
python main.py inference --image path/to/input.jpg --text "a smiling face" --checkpoint path/to/checkpoint.pth --output output.png

# Code formatting/linting (tools are in requirements.txt)
black .
flake8 .
isort .
mypy .
```

## Architecture

### Config Hierarchy (`configs/`)
Dataclass-based configuration with inheritance: `BaseConfig` → `Stage1Config` / `Stage2Config`. `BaseConfig` holds shared params (device, paths, 3DMM dimensions). `Stage1Config` adds CLIP/AES settings. `Stage2Config` adds diffusion model settings, loss weights (λ_3d, λ_id, λ_clip), and inference params (guidance_scale, num_inference_steps). `Deep3DConfig` and `PyTorch3DConfig` are standalone variants for the respective subsystems.

All paths default relative to the project root in `BaseConfig.__post_init__()`:
- `deep3d_repo_path` → `{project_root}/Deep3DFR`
- `deep3d_checkpoint_path` → `{deep3d_repo_path}/checkpoints/face_recon_model.pth`
- `bfm_model_path` → `{deep3d_repo_path}/BFM/BFM_model_front.mat`

### Core Model (`models/tcalign_diffusion.py`)
`TcAlignDiffusion` (nn.Module) is the main model integrating all submodules:
- Frozen pretrained components: `CLIPEncoder`, `Deep3DProcessor`, `PyTorch3DRenderer`, Stable Diffusion VAE
- Trainable components: `TextConditional3DEditor`, `ConditionalDiffusionMappingNetwork`
- Modified Stable Diffusion UNet (channel-expanded conv_in for multimodal input)

Two forward modes controlled by `training_mode`:
- `"3d_editor"` → `forward_3d_editor()`: returns Δθ prediction
- `"diffusion"` → `forward_diffusion()`: full pipeline with noise prediction, returns multi-component loss dict

### Key Submodules

**`TextConditional3DEditor`** (`models/text_conditional_3d_editor.py`): Takes CLIP image/text embeddings, uses AES to replace the target attribute's image subspace projection with the text's projection, then feeds the manipulated embedding through an MLP to predict Δθ.

**`AttributeEmbeddingSpace`** (`models/attribute_embedding_space.py`): Learnable per-attribute basis matrices (e.g., expression: 5 bases, pose: 4 bases). `manipulate()` projects both image and text embeddings into the attribute subspace, replaces image component with text component, keeps residual unchanged.

**`ConditionalDiffusionMappingNetwork`** (`models/conditional_diffusion_mapping_network.py`): Encodes source VAE latents (via Conv2d→pool→MLP→128d) and Δθ (MLP→128d), concatenates to 256d, fuses through MLP→512d, replicates across sequence_length, adds learned position embeddings, passes through a 4-layer TransformerEncoder, projects to condition_embed_dim (768).

**`DiffusionUNetWrapper`** (`models/diffusion_unet_wrapper.py`): Wraps Stable Diffusion's VAE + UNet. On init, expands `unet.conv_in` from 4→11 channels (4 noisy latent + 4 source latent + 3 source RGB). Supports CFG sampling via DDIM scheduler. Freezes VAE by default (`config.freeze_vae`).

### Training (`trainers/`)
`BaseTrainer` provides: mixed-precision training (AMP GradScaler), checkpoint save/load, logging, progress bars. Subclasses implement `training_step()`:

- **Stage1**: Cosine similarity loss between CLIP embedding of rendered target 3D face and target text embedding.
- **Stage2**: Loads frozen Stage 1 weights, trains in diffusion mode. During eval, generates example images via `model.generate()`.

### External Model Dependencies

This project requires several large model files that are NOT in the repo:
1. **BFM (Basel Face Model)**: `BFM_model_front.mat` — requires application at https://faces.dmi.unibas.ch/bfm/
2. **Deep3DFaceReconstruction checkpoint**: `face_recon_model.pth` — Google Drive download
3. **Stable Diffusion v1.5**: `runwayml/stable-diffusion-v1-5` — auto-downloaded by HuggingFace diffusers
4. **CLIP ViT-L/14**: `openai/clip-vit-large-patch14` — auto-downloaded by HuggingFace transformers

The `Deep3DFR/` directory contains a vendored copy of the Deep3DFaceReconstruction repo. Its modules are dynamically imported at runtime by `utils/deep3d_utils.py` via `sys.path.insert()` and `importlib`.

### Data Pipeline (`data/`)
- `FFHQDataset`: Loads images from `{data_root}/images/{split}/`, expects precomputed 3D params in `{data_root}/deep3d_params/` (`.npz` or `.npy`) and rendered images in `{data_root}/rendered/`. Creates a metadata CSV mapping images to synthetic text descriptions (random combinations of expression/pose/lighting attributes). On first run with `precompute_3d=True`, computes and caches 3D params for all training images.
- `create_transforms()`: Train split applies random horizontal flip + affine transforms + normalization to `[-1, 1]`. Val split is resize + center crop only.

### Rendering Pipeline
`Deep3DProcessor.process()` → 3D params → `PyTorch3DRenderer.render_from_params_dict()` → renders a 3D face mesh using BFM with Phong shading. The renderer uses a fixed front-facing camera at distance 10 with FOV 12.5936°.
