#!/bin/bash
#SBATCH --account=jalenj4-ic
#SBATCH --job-name=gw-regen-arr
#SBATCH --partition=secondary
#SBATCH --output=logs/regen_arr_%A_%a.out
#SBATCH --error=logs/regen_arr_%A_%a.err
#SBATCH --time=03:30:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-16
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jalen.jiang2+slurm@gmail.com

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source .venv/bin/activate
export PYTHONPATH="$SLURM_SUBMIT_DIR"

# Map task ID -> city (matches cities.yaml order)
CITIES=(arlington_tx chandler_az gilbert_az henderson_nv mesa_az tempe_az plano_tx irving_tx bellevue_wa sandy_ut beaverton_or cranberry_township_pa plymouth_mn kissimmee_fl sugar_land_tx virginia_beach_va carlsbad_ca)
CITY="${CITIES[$SLURM_ARRAY_TASK_ID]}"

echo "Task $SLURM_ARRAY_TASK_ID  City: $CITY  Start: $(date)"
python -m data_pipeline.regen_cond --config data_pipeline/cities.yaml --data data/ --city "$CITY" --workers 4
echo "End: $(date)  Exit=$?"
