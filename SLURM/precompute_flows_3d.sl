#!/bin/bash
#SBATCH --job-name      precomp3d
#SBATCH --partition=short
#SBATCH --array=0-399
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=18G
#SBATCH --output        slogs/precompute_flows_3d.%A_%a.out
#SBATCH --error         slogs/precompute_flows_3d.%A_%a.err

module purge
source /well/kir/config/modules.sh
module load Python/3.10.8-GCCcore-12.2.0
source ~/devel/venv/Python-3.10.8-GCCcore-12.2.0/omnipose_env/bin/activate

python3 -u /users/kir-fritzsche/aif490/devel/tissue_analysis/segmentation_scripts/precompute_flows_3d.py \
    --train_dir /users/kir-fritzsche/aif490/devel/tissue_analysis/segmentation_scripts/for_training/ph3/training_seeded/composites \
    --no_gpu \
    --task_id $SLURM_ARRAY_TASK_ID \
    --n_tasks $SLURM_ARRAY_TASK_COUNT
