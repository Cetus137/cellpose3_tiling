#!/bin/bash

#SBATCH --job-name      omnipose
#SBATCH --cpus-per-task 8
#SBATCH --partition=gpu_long              #Select partition. You can run sinfo command to list all partitions
#SBATCH --gpus-per-node=1                    #Number of GPUs. Always starts with 1 ( more GPU, more wait time)               
#SBATCH --mem           256G
#SBATCH --time          13:59:00         #days-minutes-seconds
#SBATCH --output        slogs/omnipose_training.%j.out
#SBATCH --error         slogs/omnipose_training.%j.err
#SBATCH --exclude       compg009,compg010,compg011,compg013

module purge
source /well/kir/config/modules.sh
module load Python/3.10.8-GCCcore-12.2.0
module load CUDA/12.0
source ~/devel/venv/Python-3.10.8-GCCcore-12.2.0/omnipose_env/bin/activate

omnipose --use_gpu --train --dir /users/kir-fritzsche/aif490/devel/tissue_analysis/segmentation_scripts/for_training/train_directory_node2 --mask_filter _masks \
         --n_epochs 4000 --pretrained_model None --learning_rate 0.1 --save_every 10 \
         --save_each  --verbose --all_channels --dim 2 \
        --batch_size 32 --diameter 0 --nclasses 3 --nchan 1