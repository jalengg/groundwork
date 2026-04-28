#!/bin/bash
#SBATCH --account=jalenj4-ic
#SBATCH --job-name=gw-regen
#SBATCH --partition=secondary
#SBATCH --output=logs/regen_%j.out
#SBATCH --error=logs/regen_%j.err
#SBATCH --time=03:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jalen.jiang2+slurm@gmail.com

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source .venv/bin/activate
export PYTHONPATH="$SLURM_SUBMIT_DIR"

echo "Start: $(date)"
python -m data_pipeline.regen_cond --config data_pipeline/cities.yaml --data data/
echo "End:   $(date)  Exit=$?"
