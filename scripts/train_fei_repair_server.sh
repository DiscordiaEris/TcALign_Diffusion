#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
if [[ -r /usr/local/lib64/libstdc++.so.6 ]]; then
    export LD_PRELOAD="/usr/local/lib64/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
fi

run_root=logs/fei_repair_20261120
mkdir -p "$run_root/stage1" "$run_root/stage2"
# One pipeline owns these new output directories. Keep historical runs intact.
exec 9>"$run_root/pipeline.lock"
flock -n 9
if [[ -e checkpoints/fei_stage1_repair_20261120/latest_checkpoint.pth ||
      -e checkpoints/fei_stage2_repair_20261120/latest_checkpoint.pth ]]; then
    echo 'Repair checkpoints already exist; resume explicitly instead of overwriting them.' >&2
    exit 1
fi
.venv/bin/python -c 'import torch; from pytorch3d import _C; assert torch.cuda.is_available()'
tar -czf "$run_root/source_snapshot.tar.gz" \
    configs data/*.py models/*.py trainers/*.py scripts/*.py scripts/train_fei_repair_server.sh \
    utils/*.py tests/*.py main.py Deep3DFR/models/*.py Deep3DFR/util/*.py
git rev-parse HEAD > "$run_root/base_commit.txt"
git diff --binary > "$run_root/working_tree.patch"

trap 'printf "failed_or_interrupted\n" > "$run_root/status"' ERR INT TERM
printf 'stage1_running\n' > "$run_root/status"
.venv/bin/python -u main.py train_stage1 --config configs/runs/fei_stage1_repair.yaml \
    >> "$run_root/stage1/console.log" 2>&1

stage1_checkpoint=checkpoints/fei_stage1_repair_20261120/final_checkpoint_epoch_49.pth
test -s "$stage1_checkpoint"
printf 'stage2_running\n' > "$run_root/status"
# Fresh SD initialization: never resume the attention-collapsed old Stage2 run.
.venv/bin/python -u main.py train_stage2 --config configs/runs/fei_stage2_repair.yaml \
    --stage1_ckpt "$stage1_checkpoint" >> "$run_root/stage2/console.log" 2>&1
printf 'complete\n' > "$run_root/status"
