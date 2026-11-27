#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export LD_PRELOAD="/usr/local/lib64/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
run_root=logs/fei_identity_refinement_20261126
checkpoint_root=checkpoints/fei_stage2_identity_refinement_20261126
mkdir -p "$run_root" "$checkpoint_root"
exec 9>"$run_root/pipeline.lock"
flock -n 9
if [[ -e "$checkpoint_root/latest_checkpoint.pth" ]]; then
    echo 'Training checkpoints already exist; resume explicitly.' >&2
    exit 1
fi
initial="${TCADIFF_INIT_CHECKPOINT:-outputs/identity_metric_repair_20261126/refinement/selected.pth}"
editor=checkpoints/fei_stage1_repair_20261120/final_checkpoint_epoch_49.pth
test -s "$initial"
test -s "$editor"
.venv/bin/python - "$initial" <<'PY'
import sys
from pathlib import Path
import torch
from utils.identity_validation import should_replace_identity_checkpoint
checkpoint = torch.load(sys.argv[1], map_location='cpu', weights_only=True)
assert checkpoint['training_config']['identity_loss_backend'] == 'arcface'
assert checkpoint['inference_config']['identity_refinement_rank'] == 2
assert checkpoint['training_config']['identity_training_alignment'] == 'generated'
assert sum('.identity_refinement.' in key for key in checkpoint['model_state_dict']) == 32
assert not checkpoint['training_config']['identity_exposure_normalization']
state = checkpoint['model_state_dict']
assert sum(key.endswith('.identity_gain_logit') for key in state) == 16
assert all(torch.isfinite(value).all() for value in state.values())
assert any(key.startswith('diffusion_unet.unet.conv_in.') for key in state)
assert any(key.startswith('diffusion_mapper.') for key in state)
assert checkpoint['inference_config']['identity_gain_only']
assert checkpoint['inference_config']['isolate_faceid_attention']
metrics = checkpoint['identity_validation']['latest']
policy = checkpoint['identity_validation']['calibration']
assert metrics['arcface_raw_valid'] == 18
assert should_replace_identity_checkpoint(metrics, None, require_expression=True,
    max_expression_error=policy['max_expression_error'],
    min_expression_progress=policy['min_expression_progress'],
    min_arcface_score=policy['min_arcface_score']), metrics
assert torch.cuda.is_available(), 'CUDA unavailable'
for file in ('models/ip-adapter-faceid/ip-adapter-faceid_sd15.bin', 'models/insightface/w600k_r50.onnx'):
    assert Path(file).is_file(), file
PY
tar -czf "$run_root/source_snapshot.tar.gz" configs data/*.py models/*.py trainers/*.py \
    scripts/*.py scripts/train_fei_identity_refinement_server.sh utils/*.py tests/*.py main.py \
    Deep3DFR/models/*.py Deep3DFR/util/*.py
git rev-parse HEAD > "$run_root/base_commit.txt"
git diff --binary > "$run_root/working_tree.patch"
printf '%s\n' "$initial" > "$run_root/initial_checkpoint.txt"
sha256sum "$initial" models/ip-adapter-faceid/ip-adapter-faceid_sd15.bin \
    models/insightface/w600k_r50.onnx datasets/face_editing_exposure_20261123/manifest.json \
    > "$run_root/input_sha256.txt"
cp "$initial" "$checkpoint_root/best_identity_checkpoint.pth"
trap 'printf "failed_or_interrupted\n" > "$run_root/status"' ERR INT TERM
printf 'identity_refinement_running\n' > "$run_root/status"
.venv/bin/python -u main.py train_stage2 --config configs/runs/fei_stage2_identity_refinement.yaml \
    --stage1_ckpt "$editor" --init_checkpoint "$initial" >> "$run_root/console.log" 2>&1
if [[ -e "$checkpoint_root/stopped_no_improvement.pth" ]]; then
    printf 'stopped_no_identity_improvement\n' > "$run_root/status"
else
    printf 'complete\n' > "$run_root/status"
fi
