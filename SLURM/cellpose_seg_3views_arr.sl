#!/bin/bash

#SBATCH --job-name      cellpose_seg
#SBATCH --cpus-per-task 1
#SBATCH --partition=gpu_short                #Select partition. You can run sinfo command to list all partitions
#SBATCH --gpus-per-node=1                    #Number of GPUs. Always starts with 1 ( more GPU, more wait time)               
#SBATCH --mem           132G
#SBATCH --time          00:45:00         #days-minutes-seconds
#SBATCH --output        slogs/cellpose_seg.%j.out
#SBATCH --error         slogs/cellpose_seg.%j.err
#SBATCH --exclude       compg009,compg010,compg011,compg013

module purge
source /well/kir/config/modules.sh
module load Python/3.10.8-GCCcore-12.2.0
module load CUDA/12.0
source ~/devel/venv/Python-3.10.8-GCCcore-12.2.0/cellpose3_env/bin/activate


T0=${SLURM_ARRAY_TASK_ID}


python3  -u /users/kir-fritzsche/aif490/devel/tissue_analysis/segmentation_scripts/tiled_segmentation_3views.py \
    --input_dir  /users/kir-fritzsche/aif490/devel/tissue_analysis/lymphnode_analysis/data2track/b2-2a_2c_pos6-01_deskew_cgt/crop3/restored     \
    --output_dir /users/kir-fritzsche/aif490/devel/tissue_analysis/lymphnode_analysis/data2track/b2-2a_2c_pos6-01_deskew_cgt/crop3/segmentation\
    --model      /users/kir-fritzsche/aif490/devel/tissue_analysis/segmentation_scripts/SLURM/models/cp3_addnoise_model\
    --gamma 1.0 \
    --tile_size 309 288 288 \
    --overlap 32 \
    --file_index ${SLURM_ARRAY_TASK_ID} \
    --verbose
##    --t_range ${T0} ${T1} \
##    --diameter None \


