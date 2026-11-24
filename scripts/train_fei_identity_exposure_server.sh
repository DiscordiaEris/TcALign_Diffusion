#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export LD_PRELOAD="/usr/local/lib64/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
run_root=logs/fei_identity_exposure_20261123
mkdir -p "$run_root"
exec 9>"$run_root/pipeline.lock"
flock -n 9
if [[ -e checkpoints/fei_stage2_identity_exposure_20261123/latest_checkpoint.pth ]]; then
    echo 'Training checkpoints already exist; resume explicitly.' >&2
    exit 1
fi
initial="${TCADIFF_INIT_CHECKPOINT:-checkpoints/fei_stage2_identity_exposure_20261123/warmstart.pth}"
editor=checkpoints/fei_stage1_repair_20261120/final_checkpoint_epoch_49.pth
test -s "$initial"
test -s "$editor"
.venv/bin/python - "$initial" <<'PY'
import json, sys
from pathlib import Path
import torch
checkpoint = torch.load(sys.argv[1], map_location='cpu', weights_only=True)
state = checkpoint['model_state_dict']
assert any(key.startswith('photometric_head.') for key in state), 'Exposure head weights required'
assert all(torch.isfinite(value).all() for value in state.values()), 'Nonfinite initial weights'
assert torch.cuda.is_available(), 'CUDA unavailable'
manifest = json.loads(Path('datasets/face_editing_exposure_20261123/manifest.json').read_text())
assert manifest['complete']
for file in ('models/ip-adapter-faceid/ip-adapter-faceid_sd15.bin', 'models/insightface/w600k_r50.onnx'):
    assert Path(file).is_file(), file
PY
tar -czf "$run_root/source_snapshot.tar.gz" configs data/*.py models/*.py trainers/*.py \
    scripts/*.py scripts/train_fei_identity_exposure_server.sh utils/*.py tests/*.py main.py \
    Deep3DFR/models/*.py Deep3DFR/util/*.py
git rev-parse HEAD > "$run_root/base_commit.txt"
git diff --binary > "$run_root/working_tree.patch"
printf '%s\n' "$initial" > "$run_root/initial_checkpoint.txt"
sha256sum "$initial" models/ip-adapter-faceid/ip-adapter-faceid_sd15.bin \
    models/insightface/w600k_r50.onnx datasets/face_editing_exposure_20261123/manifest.json \
    > "$run_root/input_sha256.txt"
trap 'printf "failed_or_interrupted\n" > "$run_root/status"' ERR INT TERM
printf 'identity_exposure_finetuning_running\n' > "$run_root/status"
.venv/bin/python -u main.py train_stage2 --config configs/runs/fei_stage2_identity_exposure.yaml \
    --stage1_ckpt "$editor" --init_checkpoint "$initial" >> "$run_root/console.log" 2>&1
printf 'complete\n' > "$run_root/status"
