#!/bin/bash
#SBATCH --account=jalenj4-ic
#SBATCH --job-name=sd15_cnet
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=18:00:00
#SBATCH --output=logs/sd15_cnet_%j.out
#SBATCH --error=logs/sd15_cnet_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jalen.jiang2+slurm@gmail.com

# SD 1.5 ControlNet trainer. Auto-resumes from latest checkpoint internally.
# All caches on /scratch to avoid /u quota.
# Submit once; re-submit after the 18h walltime — trainer picks up from checkpoint.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

OUT_DIR="${OUT_DIR:-/scratch/jalenj4/runs/sd15_cnet_v1}"
CONTROLNET_INIT="${CONTROLNET_INIT:-/scratch/jalenj4/runs/sd15_controlnet_init}"
MAX_STEPS="${MAX_STEPS:-15000}"

# Only include dirs that have cond_*.npy tiles (city dirs), excluding irving_tx holdout
CITY_DIRS=""
for d in /u/jalenj4/groundwork/data/*/; do
    city=$(basename "$d")
    [[ "$city" == "irving_tx" ]] && continue
    [[ -z "$(ls "$d"cond_*.npy 2>/dev/null | head -1)" ]] && continue
    CITY_DIRS="$CITY_DIRS $d"
done

source .venv/bin/activate
export PYTHONPATH="$SLURM_SUBMIT_DIR"
export HF_HOME=/scratch/jalenj4/hf
export HF_DATASETS_CACHE=/scratch/jalenj4/hf_datasets
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export TMPDIR=/scratch/jalenj4/tmp
mkdir -p "$TMPDIR" "$HF_HOME" "$HF_DATASETS_CACHE" logs "$OUT_DIR"

# TODO: set after running prep_sd15_dataset.py for this style, then re-submit.
# VALIDATION_ARGS='--validation_image "/u/jalenj4/groundwork/data/flux_cnet_val/val_cond_0.png" --validation_prompt "top-down raster of a US suburban road network, ..."'
VALIDATION_ARGS=""
# Note: auto-resume is handled inside train_controlnet_sd15.py — no $RESUME needed here.

echo "========================================"
echo "SD 1.5 ControlNet training"
echo "Start: $(date)   Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo unknown)"
echo "OUT_DIR=$OUT_DIR"
echo "CONTROLNET_INIT=$CONTROLNET_INIT"
echo "MAX_STEPS=$MAX_STEPS"
echo "CITY_DIRS=$CITY_DIRS"
echo "========================================"

python third_party/train_controlnet_sd15.py \
    --pretrained_model_name_or_path=runwayml/stable-diffusion-v1-5 \
    --controlnet_model_name_or_path="$CONTROLNET_INIT" \
    --output_dir="$OUT_DIR" \
    --city_dirs $CITY_DIRS \
    --resolution=512 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=8 \
    --learning_rate=1e-5 \
    --lr_warmup_steps=500 \
    --max_train_steps="$MAX_STEPS" \
    --checkpointing_steps=1000 \
    --checkpoints_total_limit=3 \
    --validation_steps=500 \
    --dataloader_num_workers=4 \
    --p_inpaint=0.5 \
    --min_road_fraction=0.05 \
    --mixed_precision=fp16 \
    --seed=42 \
    --report_to=tensorboard \
    $VALIDATION_ARGS

EXIT_CODE=$?
echo "========================================"
echo "End: $(date)"
echo "Exit: $EXIT_CODE"
echo "========================================"
