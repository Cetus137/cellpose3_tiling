#!/bin/bash
#SBATCH --job-name      omnipose
#SBATCH --account       gpu_kir.prj
#SBATCH --partition=gpu_rtx8000_48gb,gpu_v100_32gb,gpu_v100_16gb,gpu_a100_80gb,gpu_a100_40gb
#SBATCH --time=1-00:20:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-gpu=12G
#SBATCH --exclude       compg009,compg010,compg011,compg013
#SBATCH --output        slogs/omni/omnipose_training.%j.out
#SBATCH --error         slogs/omni/omnipose_training.%j.err

module purge
source /well/kir/config/modules.sh
module load Python/3.10.8-GCCcore-12.2.0
module load CUDA/12.0
source ~/devel/venv/Python-3.10.8-GCCcore-12.2.0/omnipose_env/bin/activate

python3 -u /users/kir-fritzsche/aif490/devel/tissue_analysis/segmentation_scripts/omnipose_training.py \
    --train_dir /users/kir-fritzsche/aif490/devel/tissue_analysis/segmentation_scripts/for_training/ph3/training_raw \
    --model_name omnipose_3d_ph3 \
    --n_epochs 4000 \
    --batch_size 2 \
    --learning_rate 0.005 \
    --save_every 10
