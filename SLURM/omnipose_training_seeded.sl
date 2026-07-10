#!/bin/bash
#SBATCH --job-name      omnipose_seeded
#SBATCH --account       gpu_kir.prj
#SBATCH --partition=gpu_rtx8000_48gb,gpu_v100_32gb,gpu_a100_80gb,gpu_a100_40gb
#SBATCH --time=2-00:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-gpu=256G
#SBATCH --exclude       compg009,compg010,compg011,compg013
#SBATCH --output        slogs/omni/omnipose_training_seeded.%j.out
#SBATCH --error         slogs/omni/omnipose_training_seeded.%j.err

module purge
source /well/kir/config/modules.sh
module load Python/3.10.8-GCCcore-12.2.0
module load CUDA/12.0
source ~/devel/venv/Python-3.10.8-GCCcore-12.2.0/omnipose_env/bin/activate

python3 -u /users/kir-fritzsche/aif490/devel/tissue_analysis/segmentation_scripts/omnipose_training_seeded.py \
    --train_dir /users/kir-fritzsche/aif490/devel/tissue_analysis/segmentation_scripts/for_training/ph3/training_seeded/composites \
    --model_name omnipose_3d_seeded_ph3 \
    --n_epochs 200 \
    --batch_size 4 \
    --learning_rate 0.005 \
    --save_every 10 \
    --num_workers 4
