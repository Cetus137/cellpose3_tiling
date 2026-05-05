#!/bin/bash
#SBATCH --job-name      precompute_flows
#SBATCH --partition=short
#SBATCH --array=0-199
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output        slogs/precompute_flows.%A_%a.out
#SBATCH --error         slogs/precompute_flows.%A_%a.err

module purge
source /well/kir/config/modules.sh
module load Python/3.10.8-GCCcore-12.2.0
source ~/devel/venv/Python-3.10.8-GCCcore-12.2.0/cellpose3_env/bin/activate

python3 -u /users/kir-fritzsche/aif490/devel/tissue_analysis/segmentation_scripts/precompute_flows.py \
    --train_dir /users/kir-fritzsche/aif490/devel/tissue_analysis/segmentation_scripts/for_training/ph3/training_raw/training_2D \
    --no_gpu \
    --task_id $SLURM_ARRAY_TASK_ID \
    --n_tasks $SLURM_ARRAY_TASK_COUNT