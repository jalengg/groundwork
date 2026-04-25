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

# Override via env: CFG_PROB=0.5 OUT_DIR=checkpoints/diff_cfg05 EPOCHS=150 sbatch slurm_diffusion.sh
CFG_PROB="${CFG_PROB:-0.1}"
OUT_DIR="${OUT_DIR:-checkpoints/diffusion}"
EPOCHS="${EPOCHS:-200}"

cd "$SLURM_SUBMIT_DIR"

echo "========================================"
echo "Groundwork Diffusion Training"
echo "Start Time: $(date)"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "CFG_PROB=$CFG_PROB  OUT_DIR=$OUT_DIR  EPOCHS=$EPOCHS"
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
    --vae checkpoints/vae/vae_epoch_050.pth \
    --data data/ \
    --output "$OUT_DIR" \
    --epochs "$EPOCHS" \
    --batch 4 \
    --lr 2e-5 \
    --cfg-prob "$CFG_PROB"

EXIT_CODE=$?

echo "========================================"
echo "End Time: $(date)"
echo "Exit Code: $EXIT_CODE"
echo "========================================"
