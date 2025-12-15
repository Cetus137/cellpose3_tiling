#!/bin/bash

#SBATCH --job-name      merge
#SBATCH --cpus-per-task 1
##SBATCH --partition=gpu_short                #Select partition. You can run sinfo command to list all partitions
##SBATCH --gpus-per-node=1                    #Number of GPUs. Always starts with 1 ( more GPU, more wait time)               
#SBATCH --mem           64G
#SBATCH --time          00:45:00         #days-minutes-seconds
#SBATCH --output        slogs/merge.%j.out
#SBATCH --error         slogs/merge.%j.err
#SBATCH --exclude       compg009,compg010,compg011,compg013

module purge
source /well/kir/config/modules.sh
module load Python/3.10.8-GCCcore-12.2.0
module load CUDA/12.0
source ~/devel/venv/Python-3.10.8-GCCcore-12.2.0/cellpose3_env/bin/activate



python3 /users/kir-fritzsche/aif490/devel/tissue_analysis/segmentation_scripts/merge_seg.py \
    --input      /users/kir-fritzsche/aif490/devel/tissue_analysis/lymphnode_analysis/data2track/b2-2a_2c_pos6-01_deskew_cgt/crop2/segmentation200/T0000_gamma1.00_diam30.0_masks.tif\
    --verbose \
    --output      /users/kir-fritzsche/aif490/devel/tissue_analysis/lymphnode_analysis/data2track/b2-2a_2c_pos6-01_deskew_cgt/crop2/segmentation200/T0000_gamma1.00_diam30.0_masks_merge_test.tif \
    --interface_threshold 0.10 \