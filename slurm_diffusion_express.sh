#!/bin/bash
#SBATCH --account=jalenj4-ic
#SBATCH --job-name=gw-diff-x
#SBATCH --partition=ic-express
#SBATCH --output=logs/diffusion_x_%j.out
#SBATCH --error=logs/diffusion_x_%j.err
#SBATCH --time=07:45:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:h100.1g.20gb:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jalen.jiang2+slurm@gmail.com

# Override via env: CFG_PROB=0.5 OUT_DIR=checkpoints/diff_cfg05_x EPOCHS=80 VAE_CKPT=checkpoints/vae/vae_epoch_050.pth sbatch slurm_diffusion_express.sh
CFG_PROB="${CFG_PROB:-0.5}"
OUT_DIR="${OUT_DIR:-checkpoints/diff_express}"
EPOCHS="${EPOCHS:-100}"
BATCH="${BATCH:-2}"
VAE_CKPT="${VAE_CKPT:-checkpoints/vae/vae_epoch_050.pth}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

cd "$SLURM_SUBMIT_DIR"

echo "========================================"
echo "Groundwork Diffusion Training (ic-express MIG)"
echo "Start Time: $(date)"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "CFG_PROB=$CFG_PROB  OUT_DIR=$OUT_DIR  EPOCHS=$EPOCHS  BATCH=$BATCH  VAE_CKPT=$VAE_CKPT"
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
    --lr 2e-5 \
    --cfg-prob "$CFG_PROB" \
    $EXTRA_ARGS

EXIT_CODE=$?

echo "========================================"
echo "End Time: $(date)"
echo "Exit Code: $EXIT_CODE"
echo "========================================"
