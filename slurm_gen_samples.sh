#!/bin/bash
#SBATCH --account=jalenj4-ic
#SBATCH --job-name=gw-gen-samples
#SBATCH --partition=ic-express
#SBATCH --output=logs/gen_samples_%j.out
#SBATCH --error=logs/gen_samples_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:h100.1g.20gb:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jalen.jiang2+slurm@gmail.com

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs samples
source .venv/bin/activate
export PYTHONPATH="$SLURM_SUBMIT_DIR"

VAE=checkpoints/vae_categorical/vae_epoch_050.pth

for run in diff_categorical:diffusion_epoch_150.pth diff_classwt:diffusion_epoch_150.pth diff_classwt_a100:diffusion_epoch_200.pth; do
    OUT="${run%:*}"
    CKPT="checkpoints/${OUT}/${run#*:}"
    echo "=== $OUT ==="
    python -m model.gen_samples --vae "$VAE" --diffusion "$CKPT" --out-dir "samples/$OUT" --n 4
done

echo "Done: $(date)"
