#!/bin/bash
#SBATCH --account=jalenj4-ic
#SBATCH --job-name=gw-vae-gan
#SBATCH --partition=ic-express
#SBATCH --output=logs/vae_gan_%j.out
#SBATCH --error=logs/vae_gan_%j.err
#SBATCH --time=07:45:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:h100.1g.20gb:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jalen.jiang2+slurm@gmail.com

# Override via env:
#   OUT_DIR=checkpoints/vae_gan EPOCHS=80 GAN_WARMUP=10 sbatch slurm_vae_gan.sh
OUT_DIR="${OUT_DIR:-checkpoints/vae_gan}"
EPOCHS="${EPOCHS:-80}"
BATCH="${BATCH:-2}"
GAN_WARMUP="${GAN_WARMUP:-10}"
BASE_CH="${BASE_CH:-96}"
LATENT_CH="${LATENT_CH:-4}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs "$OUT_DIR"

source .venv/bin/activate
export PYTHONPATH="$SLURM_SUBMIT_DIR"

echo "========================================"
echo "Groundwork VAE-GAN Training (ic-express MIG)"
echo "Start: $(date)"
echo "Node: $SLURM_NODELIST"
echo "OUT_DIR=$OUT_DIR  EPOCHS=$EPOCHS  BATCH=$BATCH  GAN_WARMUP=$GAN_WARMUP"
echo "BASE_CH=$BASE_CH  LATENT_CH=$LATENT_CH  EXTRA_ARGS=$EXTRA_ARGS"
echo "========================================"

python model/train_vae_gan.py \
    --data data/ \
    --output "$OUT_DIR" \
    --epochs "$EPOCHS" \
    --batch "$BATCH" \
    --gan-warmup "$GAN_WARMUP" \
    --base-ch "$BASE_CH" \
    --latent-channels "$LATENT_CH" \
    $EXTRA_ARGS

EXIT_CODE=$?
echo "End: $(date)  Exit: $EXIT_CODE"
