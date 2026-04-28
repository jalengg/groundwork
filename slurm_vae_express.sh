#!/bin/bash
#SBATCH --account=jalenj4-ic
#SBATCH --job-name=gw-vae-x
#SBATCH --partition=ic-express
#SBATCH --output=logs/vae_x_%j.out
#SBATCH --error=logs/vae_x_%j.err
#SBATCH --time=07:45:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:h100.1g.20gb:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jalen.jiang2+slurm@gmail.com

# Override via env: OUT_DIR=checkpoints/vae_paperalpha EPOCHS=50 sbatch slurm_vae_express.sh
OUT_DIR="${OUT_DIR:-checkpoints/vae}"
EPOCHS="${EPOCHS:-50}"
BATCH="${BATCH:-2}"

cd "$SLURM_SUBMIT_DIR"

echo "========================================"
echo "Groundwork VAE Training (ic-express MIG)"
echo "Start Time: $(date)"
echo "Node: $SLURM_NODELIST"
echo "OUT_DIR=$OUT_DIR  EPOCHS=$EPOCHS  BATCH=$BATCH"
echo "========================================"

mkdir -p logs "$OUT_DIR"

source .venv/bin/activate
export PYTHONPATH="$SLURM_SUBMIT_DIR"

python model/train_vae.py \
    --data data/ \
    --output "$OUT_DIR" \
    --epochs "$EPOCHS" \
    --batch "$BATCH" \
    --lr 2e-5

EXIT_CODE=$?
echo "End Time: $(date)  Exit Code: $EXIT_CODE"
