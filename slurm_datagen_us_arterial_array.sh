#!/bin/bash
#SBATCH --account=jalenj4-ic
#SBATCH --job-name=gw-arterial
#SBATCH --partition=IllinoisComputes
#SBATCH --array=0-11%1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/datagen_arterial_%A_%a.out
#SBATCH --error=logs/datagen_arterial_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jalen.jiang2+slurm@gmail.com

# One task per city, sequential (%1). 8h per city is plenty once OSM cache
# is warm. cdg.py skips existing tiles so re-runs are safe.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

source .venv/bin/activate
export PYTHONPATH="$SLURM_SUBMIT_DIR"

CITIES_CONFIG="data_pipeline/cities_us_arterial.yaml"
DATA_OUTPUT="data_arterial/"
CITY=$(python3 -c "import yaml; cfg=yaml.safe_load(open('$CITIES_CONFIG')); print(cfg['cities'][$SLURM_ARRAY_TASK_ID]['name'])")

echo "========================================"
echo "Groundwork Datagen — US Arterial"
echo "Array task $SLURM_ARRAY_TASK_ID / 11 — city: $CITY"
echo "Start: $(date)   Node: $SLURM_NODELIST"
echo "========================================"

mkdir -p logs

BEFORE=$(ls "$DATA_OUTPUT$CITY"/cond_*.npy 2>/dev/null | wc -l || true)
echo "Tiles before: $BEFORE"

python data_pipeline/cdg.py \
    --config "$CITIES_CONFIG" \
    --output "$DATA_OUTPUT" \
    --city "$CITY"

EXIT_CODE=$?

AFTER=$(ls "$DATA_OUTPUT$CITY"/cond_*.npy 2>/dev/null | wc -l || true)
echo "Tiles after: $AFTER"
echo "End: $(date)"
echo "Exit: $EXIT_CODE"
echo "========================================"
