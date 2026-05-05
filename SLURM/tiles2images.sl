#!/bin/bash

#SBATCH --job-name      tiles2images
#SBATCH --cpus-per-task 1
#SBATCH --mem           64G
#SBATCH --time          01:00:00
#SBATCH --output        slogs/tiles2images.%j.out
#SBATCH --error         slogs/tiles2images.%j.err
#SBATCH --exclude       compg009,compg010,compg011,compg013

module purge
source /well/kir/config/modules.sh
module load Python/3.10.8-GCCcore-12.2.0
source ~/devel/venv/Python-3.10.8-GCCcore-12.2.0/cellpose3_env/bin/activate

# Input/Output directories - MODIFY THESE FOR YOUR DATA
TILE_DIR="/users/kir-fritzsche/aif490/devel/tissue_analysis/lymphnode_analysis/data2track/b2-2a_2c_pos6-01_deskew_cgt/crop4/tiles_restored"
OUTPUT_DIR="/users/kir-fritzsche/aif490/devel/tissue_analysis/lymphnode_analysis/data2track/b2-2a_2c_pos6-01_deskew_cgt/crop4/tiles_restored/reconstructed"
SCRIPT_DIR="/users/kir-fritzsche/aif490/devel/tissue_analysis/segmentation_scripts"

# Reconstruction parameters
OVERLAP=32
PATTERN="*_tile_*.tif"
MARKER="timepoint_0000"

echo "Reconstructing tiles from: ${TILE_DIR}"
echo "Output directory: ${OUTPUT_DIR}"im
echo "Overlap: ${OVERLAP}"
echo "Pattern: ${PATTERN}"
echo "Marker: ${MARKER}"

python3 -u ${SCRIPT_DIR}/tiles2images.py \
    --tile_dir ${TILE_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --overlap ${OVERLAP} \
    --pattern "${PATTERN}" \
    --marker "${MARKER}" \
    --verbose

echo "Reconstruction complete!"
