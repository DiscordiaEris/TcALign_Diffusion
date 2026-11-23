#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export HF_HUB_OFFLINE=1

python_bin=.venv/bin/python
if [[ -r /usr/local/lib64/libstdc++.so.6 ]]; then
    export LD_PRELOAD="/usr/local/lib64/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
fi
"$python_bin" -c 'import torch; from pytorch3d import _C; assert torch.cuda.is_available(), "CUDA is required for the server training run"'

"$python_bin" -u main.py train_stage1 \
    --config configs/runs/fei_stage1.yaml \
    --clip_local_path models/clip-vit-large-patch14

stage1_checkpoint=checkpoints/fei_stage1/final_checkpoint_epoch_49.pth
test -s "$stage1_checkpoint"

"$python_bin" -u main.py train_stage2 \
    --config configs/runs/fei_stage2.yaml \
    --clip_local_path models/clip-vit-large-patch14 \
    --diffusion_local_path models/stable-diffusion-v1-5 \
    --stage1_ckpt "$stage1_checkpoint"
