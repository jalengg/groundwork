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

# SD 1.5 ControlNet trainer (diffusers v0.38.0). Auto-resumes from latest
# checkpoint if present. All caches on /scratch to avoid /u quota.
#
# Submit once; re-submit after the 18h walltime — $RESUME picks up.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

OUT_DIR="${OUT_DIR:-/scratch/jalenj4/runs/sd15_cnet_v1}"
CONTROLNET_INIT="${CONTROLNET_INIT:-/scratch/jalenj4/runs/sd15_controlnet_init}"
MAX_STEPS="${MAX_STEPS:-15000}"

# Build CITY_DIRS dynamically from /u/jalenj4/groundwork/data/, excluding irving_tx
CITY_DIRS=""
for d in /u/jalenj4/groundwork/data/*/; do
    city=$(basename "$d")
    [[ "$city" == "irving_tx" ]] && continue
    CITY_DIRS="$CITY_DIRS $d"
done

source .venv/bin/activate
export HF_HOME=/scratch/jalenj4/hf
export HF_DATASETS_CACHE=/scratch/jalenj4/hf_datasets
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export TMPDIR=/scratch/jalenj4/tmp
mkdir -p "$TMPDIR" "$HF_HOME" "$HF_DATASETS_CACHE" logs "$OUT_DIR"

RESUME=""
if compgen -G "$OUT_DIR/checkpoint-*" > /dev/null; then
    RESUME="--resume_from_checkpoint=latest"
    echo "Resuming from latest checkpoint."
fi

# TODO: Set VALIDATION_ARGS with --validation_image and --validation_prompt
# as needed for SD 1.5 (512x512 resolution)
VALIDATION_ARGS=""

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
    --pretrained_controlnet_model_name_or_path="$CONTROLNET_INIT" \
    --output_dir="$OUT_DIR" \
    --dataset_name=imagefolder \
    --city_dirs $CITY_DIRS \
    --image_column=image \
    --caption_column=text \
    --conditioning_image_column=conditioning_image \
    --resolution=512 \
    --train_batch_size=4 \
    --gradient_accumulation_steps=1 \
    --gradient_checkpointing \
    --use_8bit_adam \
    --learning_rate=1e-5 \
    --lr_scheduler=constant_with_warmup \
    --lr_warmup_steps=500 \
    --max_train_steps="$MAX_STEPS" \
    --checkpointing_steps=500 \
    --checkpoints_total_limit=3 \
    --validation_steps=250 \
    --proportion_empty_prompts=0.1 \
    --dataloader_num_workers=4 \
    --seed=42 \
    --report_to=tensorboard \
    $VALIDATION_ARGS \
    $RESUME

EXIT_CODE=$?
echo "========================================"
echo "End: $(date)"
echo "Exit: $EXIT_CODE"
echo "========================================"
