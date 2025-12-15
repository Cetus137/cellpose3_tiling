#!/bin/bash

#SBATCH --job-name      cellpose_training
#SBATCH --cpus-per-task 1
#SBATCH --partition=gpu_short               #Select partition. You can run sinfo command to list all partitions
#SBATCH --gpus-per-node=1                    #Number of GPUs. Always starts with 1 ( more GPU, more wait time)               
#SBATCH --mem           96G
#SBATCH --time          03:59:00         #days-minutes-seconds
#SBATCH --output        slogs/cellpose_training.%j.out
#SBATCH --error         slogs/cellpose_training.%j.err
#SBATCH --exclude       compg009,compg010,compg011,compg013

module purge
source /well/kir/config/modules.sh
module load Python/3.10.8-GCCcore-12.2.0
module load CUDA/12.0
source ~/devel/venv/Python-3.10.8-GCCcore-12.2.0/cellpose3_env/bin/activate

python3  /users/kir-fritzsche/aif490/devel/tissue_analysis/segmentation_scripts/cellpose_training.py \
