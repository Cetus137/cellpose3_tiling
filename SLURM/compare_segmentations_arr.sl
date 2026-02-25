#!/bin/bash

#SBATCH --job-name      compare_seg_arr
#SBATCH --cpus-per-task 2
#SBATCH --mem           16G
#SBATCH --time          01:00:00
#SBATCH --output        slogs/compare_seg_arr.%A_%a.out
#SBATCH --error         slogs/compare_seg_arr.%A_%a.err
#SBATCH --array         0-2
#SBATCH --exclude       compg009,compg010,compg011,compg013

module purge
source /well/kir/config/modules.sh
module load Python/3.10.8-GCCcore-12.2.0
source ~/devel/venv/Python-3.10.8-GCCcore-12.2.0/cellpose3_env/bin/activate

# =============================================================================
# CONFIGURATION - Array job for comparing single pairs in parallel
# =============================================================================

# Script directory
SCRIPT_DIR="/users/kir-fritzsche/aif490/devel/tissue_analysis/segmentation_scripts"

# IoU threshold for object matching
IOU_THRESHOLD=0.5

# Define arrays of file pairs to compare
# Add one entry per array job (modify --array parameter above to match number of pairs)
declare -a GT_FILES=(
    "/path/to/file1_masks.tif"
    "/path/to/file2_masks.tif"
    "/path/to/file3_masks.tif"
)

declare -a PRED_FILES=(
    "/path/to/file1_restored_masks.tif"
    "/path/to/file2_restored_masks.tif"
    "/path/to/file3_restored_masks.tif"
)

# Get the files for this array task
GT_FILE="${GT_FILES[$SLURM_ARRAY_TASK_ID]}"
PRED_FILE="${PRED_FILES[$SLURM_ARRAY_TASK_ID]}"

# =============================================================================
# RUN COMPARISON
# =============================================================================

echo "=============================================================================="
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "=============================================================================="
echo "Ground truth file: ${GT_FILE}"
echo "Prediction file: ${PRED_FILE}"
echo "IoU threshold: ${IOU_THRESHOLD}"
echo ""

# Run comparison
python3 -u << EOF
import sys
sys.path.insert(0, '${SCRIPT_DIR}')

from compare_segmentations import compare_segmentation_pair

# Run comparison
try:
    metrics = compare_segmentation_pair(
        file_gt='${GT_FILE}',
        file_pred='${PRED_FILE}',
        iou_threshold=${IOU_THRESHOLD},
        return_detailed=True
    )
    
    print("\n" + "="*80)
    print("Comparison complete!")
    print("="*80)
    print(f"Results saved to log file")
    
except Exception as e:
    print(f"\nERROR: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

EOF

echo ""
echo "=============================================================================="
echo "Array task ${SLURM_ARRAY_TASK_ID} finished"
echo "=============================================================================="
