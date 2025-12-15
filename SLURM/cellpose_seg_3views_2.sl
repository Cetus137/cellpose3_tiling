#!/bin/bash

#SBATCH --job-name      cellpose_seg
#SBATCH --cpus-per-task 1
#SBATCH --partition=gpu_short                #Select partition. You can run sinfo command to list all partitions
#SBATCH --gpus-per-node=1                    #Number of GPUs. Always starts with 1 ( more GPU, more wait time)               
#SBATCH --mem           48G
#SBATCH --time          03:59:00         #days-minutes-seconds
#SBATCH --output        slogs/cellpose_seg.%j.out
#SBATCH --error         slogs/cellpose_seg.%j.err
#SBATCH --exclude       compg009,compg010,compg011,compg013

module purge
source /well/kir/config/modules.sh
module load Python/3.10.8-GCCcore-12.2.0
module load CUDA/12.0
source ~/devel/venv/Python-3.10.8-GCCcore-12.2.0/cellpose3_env/bin/activate

python3  /users/kir-fritzsche/aif490/devel/tissue_analysis/segmentation_scripts/tiled_segmentation_3views.py \
    --video  /users/kir-fritzsche/aif490/devel/tissue_analysis/lymphnode_analysis/data2track/b2-2a_2c_pos6-01_deskew_cgt/crop1/restored/b2-2a_2c_pos6-01_deskew_cgt-cropped_for_segmentation_restored_3views.tif\
    --output /users/kir-fritzsche/aif490/devel/tissue_analysis/lymphnode_analysis/data2track/b2-2a_2c_pos6-01_deskew_cgt/crop1/segmentation \
    --model  /users/kir-fritzsche/aif490/devel/tissue_analysis/segmentation_scripts/models/CP_20250430_181517 \
    --gamma 1.0 \
    --tile_size 309 272 272 \
    --overlap 16 \
    ##--t_range None \