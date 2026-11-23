#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export LD_PRELOAD="/usr/local/lib64/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
run_root=logs/fei_control_20261121
mkdir -p "$run_root"
exec 9>"$run_root/pipeline.lock"
flock -n 9
if [[ -e checkpoints/fei_stage2_control_20261121/latest_checkpoint.pth ]]; then
    echo 'Control checkpoints already exist; resume explicitly.' >&2
    exit 1
fi
initial=checkpoints/fei_stage2_repair_20261120/stopped_for_control_fix_20261121.pth
editor=checkpoints/fei_stage1_repair_20261120/final_checkpoint_epoch_49.pth
test -s "$initial"
test -s "$editor"
.venv/bin/python -c 'import torch; from pytorch3d import _C; assert torch.cuda.is_available()'
tar -czf "$run_root/source_snapshot.tar.gz" configs data/*.py models/*.py \
    trainers/*.py scripts/*.py scripts/train_fei_control_server.sh utils/*.py tests/*.py \
    main.py Deep3DFR/models/*.py Deep3DFR/util/*.py
git diff --binary > "$run_root/working_tree.patch"
trap 'printf "failed_or_interrupted\n" > "$run_root/status"' ERR INT TERM
printf 'stage2_control_running\n' > "$run_root/status"
.venv/bin/python -u main.py train_stage2 --config configs/runs/fei_stage2_control.yaml \
    --stage1_ckpt "$editor" --init_checkpoint "$initial" >> "$run_root/console.log" 2>&1
printf 'complete\n' > "$run_root/status"
