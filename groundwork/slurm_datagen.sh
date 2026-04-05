#!/bin/bash
#SBATCH --account=jalenj4-ic
#SBATCH --job-name=gw-datagen
#SBATCH --partition=secondary
#SBATCH --output=logs/datagen_%j.out
#SBATCH --error=logs/datagen_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jalen.jiang2+slurm@gmail.com

cd "$SLURM_SUBMIT_DIR"

echo "========================================"
echo "Groundwork Data Generation"
echo "Start Time: $(date)"
echo "Node: $SLURM_NODELIST"
echo "========================================"

mkdir -p logs

source .venv/bin/activate
export PYTHONPATH="$SLURM_SUBMIT_DIR"

echo "Tile counts before:"
for d in data/*/; do
    city=$(basename "$d")
    count=$(ls "$d"/cond_*.npy 2>/dev/null | wc -l)
    echo "  $city: $count"
done

echo "========================================"
echo "Starting data generation (skips existing tiles)..."
python data_pipeline/cdg.py --config data_pipeline/cities.yaml --output data/

EXIT_CODE=$?

echo "========================================"
echo "Tile counts after:"
for d in data/*/; do
    city=$(basename "$d")
    count=$(ls "$d"/cond_*.npy 2>/dev/null | wc -l)
    echo "  $city: $count"
done

echo "========================================"
echo "End Time: $(date)"
echo "Exit Code: $EXIT_CODE"
echo "========================================"
