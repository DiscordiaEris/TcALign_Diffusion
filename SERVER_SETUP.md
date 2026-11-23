# Paired face editing on the server

TcADiff trains the text-conditioned 3D editor first. Stage 2 uses the frozen
editor's predicted 3D change during training and inference. The diffusion
mapper uses separate expression, pose, and light embeddings to attend to
spatial source-image features, then adapts the Stable Diffusion text tokens.
The UNet denoises real target-photo latents and conditions on the source latent
and RGB image. Its auxiliary 3D loss compares the generated and target
portraits using the same target-landmark alignment.

## Data

The trainer expects same-identity photo pairs that implement the instruction.
The CSV is UTF-8 and the image paths are relative to the split's image folder.

```text
datasets/face_editing/
  images/train/
  images/val/
  metadata_train.csv
  metadata_val.csv
```

```csv
source,target,text
source_0001.jpg,target_0001.jpg,Make the person smile.
```

Use the original color [FEI Face Database](https://fei.edu.br/~cet/facedatabase.html)
for the initial run. The [downloaded mirror](https://www.kaggle.com/datasets/whizzkid/fei-face-data)
contains all 2,800 original photographs. Convert it with:

```bash
.venv/bin/python -m scripts.prepare_fei datasets/fei_raw/fei-face-data.zip
```

The script groups captures by subject and by recording condition. The default
split reserves 20 of 200 identities for validation. It covers frontal and
moderate left/right head turns, neutral and smiling faces, and three
illumination levels. FEI's lab backgrounds and limited attribute labels restrict
what this first run can demonstrate on in-the-wild portraits, other
expressions, and directional lighting. During first use, the loader caches
Deep3D parameters, aligned transforms, and BFM renders. It logs failed face
detections alongside the cache without modifying the CSV.

## Environment

The server run uses CUDA 11.8, PyTorch 2.5.1, torchvision 0.20.1, diffusers
0.32.2, transformers 4.48.3, OpenCV 4.10 with YuNet, and PyTorch3D 0.7.9 built
against the same CUDA and PyTorch versions. The local model files are:

- `models/clip-vit-large-patch14/`
- `models/stable-diffusion-v1-5/` with UNet, VAE, text encoder, and tokenizer.
- `models/face_detection_yunet_2023mar.onnx` from the
  [OpenCV model zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet).
- `Deep3DFR/BFM/BFM_model_front.mat`
- `Deep3DFR/checkpoints/face_recon_model.pth`

On T7920-Server, run commands outside the Codex sandbox for GPU access. The
training script preloads the host's newer libstdc++ for the compiled PyTorch3D
extension.

## Training and inference

```bash
cd /home/HRTao/TcADiff
tmux -L tcadiff-fei new-session -d -s fei-training \
  'bash scripts/train_fei_server.sh >> logs/fei_20261113/training.log 2>&1'
tail -f logs/fei_20261113/training.log
```

The script runs Stage 1 for 50 epochs and starts Stage 2 for 100 epochs after
the Stage 1 final checkpoint is written. Periodic checkpoints roll over to
`latest_checkpoint.pth` in each stage's directory. They contain the learned
weights and optimizer state. Inference loads the frozen base models from the
paths used during training. The earlier broadcast-residual mapper checkpoints
are incompatible with the spatial attention mapper.

```bash
LD_PRELOAD=/usr/local/lib64/libstdc++.so.6 .venv/bin/python main.py inference \
  --image /path/to/input.jpg --text 'Make the person smile.' \
  --checkpoint checkpoints/fei_stage2/latest_checkpoint.pth \
  --clip_local_path models/clip-vit-large-patch14 \
  --diffusion_local_path models/stable-diffusion-v1-5 \
  --strength 0.65 --output outputs/edited.png
```

## Evaluate a trained checkpoint

The training log records validation denoising loss over the validation set.
Epoch-end SSIM and PSNR come from the first validation batch. Run the
separate evaluation command for per-pair results and full-set averages:

```bash
LD_PRELOAD=/usr/local/lib64/libstdc++.so.6 \
TORCH_HOME="$PWD/.venv/torch-cache" \
.venv/bin/python main.py evaluate \
  --checkpoint checkpoints/fei_stage2/final_checkpoint_epoch_99.pth \
  --clip_local_path models/clip-vit-large-patch14 \
  --diffusion_local_path models/stable-diffusion-v1-5 \
  --sface models/face_recognition_sface_2021dec.onnx \
  --output outputs/fei_evaluation/full
```

By default this evaluates all usable validation pairs with the inference
configuration's 50 DDIM steps and CFG scale of 7.5. The output contains
`pairs.csv`, `summary.json` with sample counts and averages per attribute
and instruction, and example triptychs in source, target, generated order.

- Pose metrics compare the estimated yaw against the real paired photo
  in degrees and measure change in the expected direction. Left/right
  prompts also report whether the generated face points to the requested
  side of the image by at least 5°. Detection failures count as misses
  for rates and are excluded from angular error, whose denominator is reported.
- Expression and lighting errors are mean absolute differences in Deep3D
  coefficient space. Drift of unrequested attributes is measured against the
  source. These scores inherit Deep3D's estimator error.
- SSIM, PSNR and optional LPIPS compare with the real target photo and
  include the unchanged source as a comparison. Optional SFace reports
  source-to-generated cosine and source-to-real-target cosine using the
  [OpenCV SFace weights](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface).
  Its valid-pair count can differ from the Deep3D face detection count.
- Use `--instruction 'toward the left side'` to focus on left poses.
  Compare `--condition predicted`, `--condition oracle` (the paired target's
  measured 3D change), and `--condition text-only`. Keep seed, pair selection
  and sampling settings fixed. Oracle uses target-photo information and is
  only a diagnostic. Use `--strength` and `--guidance` to test sampling tradeoffs.

To include LPIPS, add `--lpips`. It requires the optional `lpips` package and torchvision's
`alexnet-owt-7be5be79.pth` in the torch hub checkpoint cache.

## Stage 2 with spatial geometry and identity supervision

The run configuration `configs/runs/fei_stage2_geometry_identity.yaml`
uses the completed Stage 1 editor and starts Stage 2 from the local pretrained
Stable Diffusion weights. The geometry adapter adds learned UNet residuals
from target BFM RGB, surface normals and visibility, warped into the source
portrait's frame using its cached alignment. Half of training edits
use measured target coefficients for the requested attributes. The other
half use the editor's predicted coefficients, which are also used for
validation and inference. A frozen FaceNet VGGFace2 encoder adds an identity
loss on generated crops aligned with the paired portrait. SFace remains
available for independent evaluation.

Install `facenet-pytorch==2.6.0` with `--no-deps` in the existing CUDA
environment and place the upstream `20180402-114759-vggface2.pt` checkpoint at
`models/facenet/vggface2.pt`. The package is used for its pretrained
InceptionResnetV1 and should use the existing PyTorch installation.

```bash
LD_PRELOAD=/usr/local/lib64/libstdc++.so.6 HF_HUB_OFFLINE=1 \
.venv/bin/python -u main.py train_stage2 \
  --config configs/runs/fei_stage2_geometry_identity.yaml \
  --stage1_ckpt checkpoints/fei_stage1/final_checkpoint_epoch_49.pth \
  --clip_local_path models/clip-vit-large-patch14 \
  --diffusion_local_path models/stable-diffusion-v1-5
```

Inference and evaluation detect the new geometry adapter from the checkpoint.
For a controlled comparison, pass the same `--steps`, `--strength`,
`--guidance`, and evaluation seed to both runs. Check identity and
instruction metrics separately. The geometry/identity run stores its
checkpoints, logs, and validation previews in separate directories.

## Repaired training run (2026-11-20)

The old Adam optimizer coupled L2 decay to the adaptive update and severely
attenuated pretrained cross-attention weights. The repair uses AdamW, excludes
biases/norm vectors from decay, and gives pretrained UNet attention LR `1e-5`
while new source/geometry/mapper paths use `1e-4`. Do not resume the old Stage2
optimizer/weights into this run. Stage1 is retrained with equal instruction
sampling and semantic-preserving training paraphrases; validation text and
same-identity metadata stay unchanged. Stage2 uses predicted geometry in both
training and inference, removing the old 50% oracle-geometry mixture.

```bash
tmux -L tcadiff-repair new-session -d -s fei-repair -c /home/HRTao/TcADiff \
  'bash scripts/train_fei_repair_server.sh >> logs/fei_repair_20261120/pipeline.log 2>&1'
```

Create `logs/fei_repair_20261120/` before redirecting there. The script locks the
run, rejects overwriting existing repair checkpoints, archives source/configs,
and trains Stage1 for 50 epochs then Stage2 for 100 epochs. Outputs are isolated
under `*_repair_20261120` directories. The `status` file records the active stage.
Use `tmux -L tcadiff-repair attach -t fei-repair` to inspect the session.

Validation uses a fixed noise/timestep stream, restores training RNG state,
and reports each loss component with its valid count. Sampled images additionally
log identity cosine, requested attribute errors, unrequested attribute drift,
and the unchanged-source baseline. These aligned-estimator scores are proxies,
not an independent face-detection success measure. Check full holdout results
with `main.py evaluate` as well as the small preview.

New checkpoints store inference strength/CFG/steps/BF16 defaults. The repair
uses strength `0.65`, CFG `1.5`, and 50 steps; explicit CLI options override them.
Reduced strength is an identity safeguard, not proof that editing works.
The bounded GPU check is `python -m scripts.smoke_training_repair`; it exercises
both stages, save/reload, sampled metrics, repeatable validation, and attention
norm retention. Unit tests require the same libstdc++ preload as training:

```bash
LD_PRELOAD=/usr/local/lib64/libstdc++.so.6 OMP_NUM_THREADS=2 \
.venv/bin/python -m unittest discover -s tests -v
```

Diagnostic evidence is in `outputs/repair_diagnostics_20261120/` and
`outputs/diagnostics/20261120_stage1/`. A successful smoke test establishes
execution correctness; visual and semantic quality require the new training
and holdout evaluation.

## Source-start control repair (2026-11-21)

The previous img2img run mostly copied source pose at strength .65 even with
oracle geometry. Pure-noise sampling obeyed pose instructions but changed
identity. `configs/runs/fei_stage2_control.yaml` adds a supervised source-start
branch mixed with ordinary target-noised diffusion. On conditional source
examples at t250..700, the regression target is
`epsilon + sqrt(alpha/(1-alpha)) * (source_latent-target_latent)` so that x0
recovery targets the edited portrait. CFG-dropped examples retain ordinary
noising. This is an empirical transport objective; single-step algebra alone
does not guarantee iterative sampling quality.

Targets are registered to the source landmark frame (`At @ inverse(As)`),
removing the approximately45–61px face-frame shifts in pose photos. Only
requested 3D attributes receive displacement-normalized control loss; lighting
also receives direct aligned-face luminance supervision because better Deep3D
lighting coefficients did not ensure visibly dimmer pixels. The original
metadata/caches remain unchanged. Defaults keep older configurations on their
original objectives. Checkpoints save the registered target convention, which
standalone evaluation restores before loading data.

```bash
mkdir -p logs/fei_control_20261121
tmux -L tcadiff-control new-session -d -s fei-control -c /home/HRTao/TcADiff \
  'bash scripts/train_fei_control_server.sh >> logs/fei_control_20261121/pipeline.log 2>&1'
```

The launcher reuses the completed repaired Stage1, initializes Stage2 weights
from `stopped_for_control_fix_20261121.pth`, and resets optimizer/epoch/step for
the new objective. It does not initialize from the deliberately small-subset
short-training experiment. Logs/checkpoints/outputs live in separate
`fei_*control*20261121` directories, with a lock and source snapshot.

Diagnostic scripts, before/after outputs, and reports are preserved under
`outputs/control_fix_20261121/`. Compare same source and the same random seed
across instructions; pose success, identity, exposure and expression should
be reported separately. Short-training success proves a controllable pathway,
not full validation-set quality or unrestricted text generalization.
