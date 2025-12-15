#!/bin/bash

#SBATCH --job-name      omni_seg
#SBATCH --cpus-per-task 1
#SBATCH --partition=gpu_short                #Select partition. You can run sinfo command to list all partitions
#SBATCH --gpus-per-node=1                    #Number of GPUs. Always starts with 1 ( more GPU, more wait time)               
#SBATCH --mem           196G
#SBATCH --time          03:15:00         #days-minutes-seconds
#SBATCH --output        slogs/omnipose_seg.%j.out
#SBATCH --error         slogs/omnipose_seg.%j.err
#SBATCH --exclude       compg009,compg010,compg011,compg013

module purge
source /well/kir/config/modules.sh
module load Python/3.10.8-GCCcore-12.2.0
module load CUDA/12.0
source ~/devel/venv/Python-3.10.8-GCCcore-12.2.0/omnipose_env/bin/activate


python3  /users/kir-fritzsche/aif490/devel/tissue_analysis/segmentation_scripts/tiled_segmentation_3views_omnipose.py \
    --video  /users/kir-fritzsche/aif490/devel/tissue_analysis/lymphnode_analysis/data2track/b2-2a_2c_pos6-01_deskew_cgt/crop1/restored200/restored_timepoint_0000.tif\
    --output /users/kir-fritzsche/aif490/devel/tissue_analysis/lymphnode_analysis/data2track/b2-2a_2c_pos6-01_deskew_cgt/crop1/omni_seg \
    --model  /users/kir-fritzsche/aif490/devel/tissue_analysis/segmentation_scripts/for_training/train_directory_node1/models/cellpose_residual_on_style_on_concatenation_off_omni_abstract_nclasses_3_nchan_1_dim_2_train_directory_node1_2025_12_05_10_57_16.727690_epoch_40 \
    --gamma 1.0 \
    --tile_size 309 288 288 \
    --overlap 32 \



