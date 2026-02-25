#!/bin/bash

#SBATCH --job-name      batch_dir_recon
#SBATCH --cpus-per-task 4
#SBATCH --mem           10G
#SBATCH --time          00:10:00
#SBATCH --output        slogs/batch_dir_recon.%j.out
#SBATCH --error         slogs/batch_dir_recon.%j.err
#SBATCH --exclude       compg009,compg010,compg011,compg013

module purge
source /well/kir/config/modules.sh
module load Python/3.10.8-GCCcore-12.2.0
module load CUDA/12.0
source ~/devel/venv/Python-3.10.8-GCCcore-12.2.0/cellpose4_env/bin/activate

file_index=${SLURM_ARRAY_TASK_ID}
echo "Processing file index: ${file_index}"

python3 -u /users/kir-fritzsche/aif490/devel/tissue_analysis/segmentation_scripts/batch_dir_reconstruction.py \
    --input_dir  /users/kir-fritzsche/aif490/devel/tissue_analysis/cellpose/data_live_node1_3d/segmented \
    --output_dir /users/kir-fritzsche/aif490/devel/tissue_analysis/cellpose/data_live_node1_3d/segmented/reconstructed \
    --min_size 5000 \
    --cellprob_threshold 0.0 \
    --do_3D True \
    --file_index ${file_index} \
    --verbose
