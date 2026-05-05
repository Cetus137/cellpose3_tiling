#!/bin/bash

#SBATCH --job-name      cellpose_training
#SBATCH --account gpu_kir.prj
#SBATCH --cpus-per-task 1
#SBATCH --partition=gpu_rtx8000_48gb,gpu_v100_32gb,gpu_v100_16gb,gpu_a100_80gb,gpu_a100_40gb
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-gpu=160G
#SBATCH --time          01-12:00:00
#SBATCH --output        slogs/cellpose_training.%j.out
#SBATCH --error         slogs/cellpose_training.%j.err
#SBATCH --exclude       compg009,compg010,compg011,compg013

module purge
source /well/kir/config/modules.sh
module load Python/3.10.8-GCCcore-12.2.0
module load CUDA/12.0
source ~/devel/venv/Python-3.10.8-GCCcore-12.2.0/cellpose3_env/bin/activate

python3 -u /users/kir-fritzsche/aif490/devel/tissue_analysis/segmentation_scripts/cellpose_training.py \
    --train_dir /users/kir-fritzsche/aif490/devel/tissue_analysis/segmentation_scripts/for_training/ph3/training_raw/training_2D \
    --max_tiles 30000 \
    --seed 42 \
    --min_label_pixels 5
