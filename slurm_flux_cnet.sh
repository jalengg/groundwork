#!/bin/bash
#SBATCH --account=jalenj4-ic
#SBATCH --job-name=flux_cnet
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=18:00:00
#SBATCH --output=logs/flux_cnet_%j.out
#SBATCH --error=logs/flux_cnet_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jalen.jiang2+slurm@gmail.com

# Flux ControlNet training driver. Auto-resumes from latest checkpoint if present.
# Override OUT_DIR / CONFIG via env if you want.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

CONFIG="${CONFIG:-config/flux_cnet.json}"
OUT_DIR="${OUT_DIR:-runs/flux_cnet_v1}"

source SimpleTuner/.venv/bin/activate
export HF_HOME=/u/jalenj4/.cache/hf
export TOKENIZERS_PARALLELISM=false
export NCCL_P2P_DISABLE=1
export PYTHONPATH="$SLURM_SUBMIT_DIR/SimpleTuner"

mkdir -p logs "$OUT_DIR"

echo "========================================"
echo "Flux ControlNet training"
echo "Start: $(date)"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)"
echo "Config: $CONFIG"
echo "OUT_DIR: $OUT_DIR"
echo "========================================"

RESUME=""
if compgen -G "$OUT_DIR/checkpoint-*" > /dev/null; then
  RESUME="--resume_from_checkpoint=latest"
  echo "Resuming from latest checkpoint."
fi

cd SimpleTuner
accelerate launch --num_processes 1 --mixed_precision bf16 \
    train.py \
    --config "$SLURM_SUBMIT_DIR/$CONFIG" \
    $RESUME

EXIT_CODE=$?
echo "========================================"
echo "End: $(date)"
echo "Exit: $EXIT_CODE"
echo "========================================"
