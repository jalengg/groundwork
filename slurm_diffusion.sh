#!/bin/bash
#SBATCH --account=jalenj4-ic
#SBATCH --job-name=gw-diff
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --output=logs/diffusion_%j.out
#SBATCH --error=logs/diffusion_%j.err
#SBATCH --time=18:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:A100:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jalen.jiang2+slurm@gmail.com

# Override via env, e.g.:
#   EXTRA_ARGS='--class-weights 1.0,1.2,1.4,1.4,1.4 --local-module lde' \
#   OUT_DIR=checkpoints/diff_eq9_planAfix EPOCHS=200 sbatch slurm_diffusion.sh
CFG_PROB="${CFG_PROB:-0.1}"
OUT_DIR="${OUT_DIR:-checkpoints/diffusion}"
EPOCHS="${EPOCHS:-200}"
BATCH="${BATCH:-4}"
LR="${LR:-2e-5}"
VAE_CKPT="${VAE_CKPT:-checkpoints/vae_categorical/vae_epoch_050.pth}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

cd "$SLURM_SUBMIT_DIR"

echo "========================================"
echo "Groundwork Diffusion Training"
echo "Start Time: $(date)"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "CFG_PROB=$CFG_PROB  OUT_DIR=$OUT_DIR  EPOCHS=$EPOCHS  BATCH=$BATCH  LR=$LR"
echo "VAE_CKPT=$VAE_CKPT"
echo "EXTRA_ARGS=$EXTRA_ARGS"
echo "========================================"

mkdir -p logs "$OUT_DIR"

source .venv/bin/activate
export PYTHONPATH="$SLURM_SUBMIT_DIR"

echo "Dataset:"
total=0
for d in data/*/; do
    city=$(basename "$d")
    count=$(ls "$d"cond_*.npy 2>/dev/null | wc -l)
    total=$((total + count))
    echo "  $city: $count"
done
echo "  TOTAL: $total tiles"

echo "========================================"

python model/train_diffusion.py \
    --vae "$VAE_CKPT" \
    --data data/ \
    --output "$OUT_DIR" \
    --epochs "$EPOCHS" \
    --batch "$BATCH" \
    --lr "$LR" \
    --cfg-prob "$CFG_PROB" \
    $EXTRA_ARGS

EXIT_CODE=$?

echo "========================================"
echo "End Time: $(date)"
echo "Exit Code: $EXIT_CODE"
echo "========================================"
