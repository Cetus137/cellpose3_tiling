#!/bin/bash

#SBATCH --job-name      batch_dir_recon
#SBATCH --cpus-per-task 1
#SBATCH --mem           4G
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
    --input_dir  /users/kir-fritzsche/aif490/devel/tissue_analysis/lymphnode_analysis/data2validate/crop1_ordered/phalloidin_seg/ \
    --output_dir /users/kir-fritzsche/aif490/devel/tissue_analysis/lymphnode_analysis/data2validate/crop1_ordered/phalloidin_seg/reconstructed_cellprob_minus2\
    --min_size 5000 \
    --cellprob_threshold -2.0 \
    --do_3D True \
    --file_index ${file_index} \
    --verbose
