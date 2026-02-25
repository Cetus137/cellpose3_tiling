#!/bin/bash

#SBATCH --job-name      threshold_cellprob
#SBATCH --cpus-per-task 1
#SBATCH --mem           2G
#SBATCH --time          00:30:00
#SBATCH --output        slogs/threshold_cellprob.%j.out
#SBATCH --error         slogs/threshold_cellprob.%j.err
#SBATCH --exclude       compg009,compg010,compg011,compg013

module purge
source /well/kir/config/modules.sh
module load Python/3.10.8-GCCcore-12.2.0
source ~/devel/venv/Python-3.10.8-GCCcore-12.2.0/cellpose3_env/bin/activate

# Input/Output directories - MODIFY THESE FOR YOUR DATA
INPUT_DIR="/users/kir-fritzsche/aif490/devel/tissue_analysis/lymphnode_analysis/data2track/b2-6a_overview_pos1-01_deskew_cgt/crop2/boundary/reconstructed"
BASE_OUTPUT_DIR="/users/kir-fritzsche/aif490/devel/tissue_analysis/lymphnode_analysis/data2track/b2-6a_overview_pos1-01_deskew_cgt/crop2/boundary/reconstructed_thresholded"
SCRIPT_DIR="/users/kir-fritzsche/aif490/devel/tissue_analysis/segmentation_scripts"

# Thresholding parameters
THRESHOLD=115
PATTERN="*cell_prob*.tif"

# Create output directory name with threshold
OUTPUT_DIR="${BASE_OUTPUT_DIR}_prob_thresh_${THRESHOLD}"

echo "Input directory: ${INPUT_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Threshold: ${THRESHOLD}"
echo "Pattern: ${PATTERN}"

python3 -u ${SCRIPT_DIR}/threshold_cellprob.py \
    --input_dir ${INPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --threshold ${THRESHOLD} \
    --pattern "${PATTERN}" \
    --verbose

echo "Thresholding complete!"
