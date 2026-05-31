#!/bin/bash
#SBATCH --account=jalenj4-ic
#SBATCH --job-name=sdxl_cnet_arterial
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=18:00:00
#SBATCH --output=logs/sdxl_cnet_arterial_%j.out
#SBATCH --error=logs/sdxl_cnet_arterial_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jalen.jiang2+slurm@gmail.com

# SDXL ControlNet trainer for US arterial grid style (warm-start from us_suburbs).
# Diffusers v0.38.0. Auto-resumes from latest checkpoint if present.
# All caches on /scratch to avoid /u quota.
#
# Submit once; re-submit after the 18h walltime — $RESUME picks up.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

OUT_DIR="${OUT_DIR:-/scratch/jalenj4/runs/sdxl_cnet_us_arterial_v1}"
DATA_DIR="${DATA_DIR:-/u/jalenj4/groundwork/data/flux_cnet_us_arterial_hf}"
MAX_STEPS="${MAX_STEPS:-25000}"

source .venv/bin/activate
export HF_HOME=/scratch/jalenj4/hf
export HF_DATASETS_CACHE=/scratch/jalenj4/hf_datasets
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false
export NCCL_P2P_DISABLE=1
export PYTHONUNBUFFERED=1
export TMPDIR=/scratch/jalenj4/tmp
mkdir -p "$TMPDIR" "$HF_HOME" "$HF_DATASETS_CACHE" logs "$OUT_DIR"

RESUME=""
if compgen -G "$OUT_DIR/checkpoint-*" > /dev/null; then
    RESUME="--resume_from_checkpoint=latest"
    echo "Resuming from latest checkpoint."
fi

echo "========================================"
echo "SDXL ControlNet training — US arterial grid"
echo "Start: $(date)   Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo unknown)"
echo "OUT_DIR=$OUT_DIR"
echo "DATA_DIR=$DATA_DIR"
echo "MAX_STEPS=$MAX_STEPS"
echo "========================================"

accelerate launch --num_processes 1 --mixed_precision fp16 \
    third_party/train_controlnet_sdxl.py \
    --pretrained_model_name_or_path=stabilityai/stable-diffusion-xl-base-1.0 \
    --pretrained_vae_model_name_or_path=madebyollin/sdxl-vae-fp16-fix \
    --controlnet_model_name_or_path=jalengg/groundwork-sdxl-cnet-us-suburbs \
    --output_dir="$OUT_DIR" \
    --dataset_name=imagefolder \
    --train_data_dir="$DATA_DIR" \
    --image_column=image \
    --caption_column=text \
    --conditioning_image_column=conditioning_image \
    --resolution=1024 \
    --mixed_precision=fp16 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=8 \
    --gradient_checkpointing \
    --use_8bit_adam \
    --learning_rate=1e-5 \
    --lr_scheduler=constant_with_warmup \
    --lr_warmup_steps=500 \
    --max_train_steps="$MAX_STEPS" \
    --checkpointing_steps=1000 \
    --checkpoints_total_limit=3 \
    --validation_steps=500 \
    --proportion_empty_prompts=0.1 \
    --dataloader_num_workers=4 \
    --seed=42 \
    --report_to=tensorboard \
    $RESUME

# TODO: fill in after running prep_flux_dataset.py for this style
# --validation_image "/u/jalenj4/groundwork/data/flux_cnet_us_arterial_val/val_cond_0.png" "/u/jalenj4/groundwork/data/flux_cnet_us_arterial_val/val_cond_1.png" \
# --validation_prompt "top-down satellite-style raster of an American inner suburban arterial grid road network, high-contrast color-coded road class map, flat color, vector style, no texture, no shading" "top-down satellite-style raster of an American inner suburban arterial grid road network, high-contrast color-coded road class map, flat color, vector style, no texture, no shading" \

EXIT_CODE=$?
echo "========================================"
echo "End: $(date)"
echo "Exit: $EXIT_CODE"
echo "========================================"
