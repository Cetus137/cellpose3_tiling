#!/bin/bash
#SBATCH --job-name      omnipose
#SBATCH --time=00:39:00
#SBATCH --cpus-per-task 4
#SBATCH --partition=gpu_short
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-gpu=15G
#SBATCH --constraint="a100|rtx8000|"
#SBATCH --exclude       compg009,compg010,compg011,compg013
#SBATCH --output        slogs/omnipose_training.%j.out
#SBATCH --error         slogs/omnipose_training.%j.err

module purge
source /well/kir/config/modules.sh
module load Python/3.10.8-GCCcore-12.2.0
module load CUDA/12.0
source ~/devel/venv/Python-3.10.8-GCCcore-12.2.0/omnipose_env/bin/activate

omnipose --use_gpu --train --dir /users/kir-fritzsche/aif490/devel/tissue_analysis/cellpose/data/for_training --mask_filter _masks \
         --n_epochs 4000 --pretrained_model None --learning_rate 0.05 --save_every 10 \
         --save_each  --verbose --dim 3 \
        --batch_size 1 --diameter 0 --nclasses 3 --nchan 1