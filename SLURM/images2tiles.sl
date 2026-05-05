#!/bin/bash
#SBATCH --job-name     images2tiles
#SBATCH --cpus-per-task 1
##SBATCH --partition=short                #Select partition. You can run sinfo command to list all partitions
##SBATCH --gpus-per-node=1                    #Number of GPUs. Always starts with 1 ( more GPU, more wait time)               
#SBATCH --mem           256G
#SBATCH --time          01:30:00         #days-minutes-seconds
#SBATCH --output        slogs/img2tiles.%j.out
#SBATCH --error         slogs/img2tiles.%j.err
#SBATCH --exclude       compg009,compg010,compg011,compg013

module purge
source /well/kir/config/modules.sh
module load Python/3.10.8-GCCcore-12.2.0
module load CUDA/12.0
source ~/devel/venv/Python-3.10.8-GCCcore-12.2.0/cellpose4_env/bin/activate

file_path='/users/kir-fritzsche/aif490/devel/tissue_analysis/segmentation_scripts/for_training/ph3/ph3_vol3_z50-306_y0-2032_x0-5838.tif'


python3 -u /users/kir-fritzsche/aif490/devel/tissue_analysis/segmentation_scripts/image2tiles.py \
    --output_dir /users/kir-fritzsche/aif490/devel/tissue_analysis/segmentation_scripts/for_training/ph3/tiles \
    --file_path ${file_path} \
    --tile_size 256 256 256 \
    --verbose