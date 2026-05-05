#!/bin/bash
#SBATCH --job-name      prep_2d
#SBATCH --partition=short
#SBATCH --array=0-99
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output        slogs/prep_2d.%A_%a.out
#SBATCH --error         slogs/prep_2d.%A_%a.err

module purge
source /well/kir/config/modules.sh
module load Python/3.10.8-GCCcore-12.2.0
source ~/devel/venv/Python-3.10.8-GCCcore-12.2.0/cellpose3_env/bin/activate

python3 -u /users/kir-fritzsche/aif490/devel/tissue_analysis/segmentation_scripts/prepare_2d_training.py \
    --train_dir /users/kir-fritzsche/aif490/devel/tissue_analysis/segmentation_scripts/for_training/ph3/training_raw \
    --task_id $SLURM_ARRAY_TASK_ID \
    --n_tasks $SLURM_ARRAY_TASK_COUNT

