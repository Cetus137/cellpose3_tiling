#!/bin/bash
#SBATCH --job-name      cpsam_train
#SBATCH --account       gpu_kir.prj
#SBATCH --partition=gpu_rtx8000_48gb,gpu_v100_32gb,gpu_a100_80gb
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-gpu=40G
#SBATCH --time=2-00:00:00
#SBATCH --output        slogs/cpSAM_training.%j.out
#SBATCH --error         slogs/cpSAM_training.%j.err
#SBATCH --exclude       compg009,compg010,compg011,compg013

module purge
source /well/kir/config/modules.sh
module load Python/3.10.8-GCCcore-12.2.0
module load CUDA/12.0
source ~/devel/venv/Python-3.10.8-GCCcore-12.2.0/cellpose4_env/bin/activate

python3 -u /users/kir-fritzsche/aif490/devel/tissue_analysis/segmentation_scripts/cpSAM_training.py \
    --train_dir /users/kir-fritzsche/aif490/devel/tissue_analysis/segmentation_scripts/for_training/ph3/training_raw/training_2D \
    --max_tiles 50000 \
    --min_label_pixels 5 \
    --n_epochs 500 \
    --batch_size 4 \
    --learning_rate 5e-5 \
    --model_name cpsam_ph3
