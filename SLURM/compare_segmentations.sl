#!/bin/bash

#SBATCH --job-name      compare_seg
#SBATCH --cpus-per-task 4
#SBATCH --mem           32G
#SBATCH --time          02:00:00
#SBATCH --output        slogs/compare_seg.%j.out
#SBATCH --error         slogs/compare_seg.%j.err
#SBATCH --exclude       compg009,compg010,compg011,compg013

module purge
source /well/kir/config/modules.sh
module load Python/3.10.8-GCCcore-12.2.0
source ~/devel/venv/Python-3.10.8-GCCcore-12.2.0/cellpose3_env/bin/activate

# =============================================================================
# CONFIGURATION - MODIFY THESE FOR YOUR DATA
# =============================================================================

# Directories containing segmentations to compare
GT_DIR="/path/to/ground_truth_segmentations"
PRED_DIR="/path/to/predicted_segmentations"

# File patterns for matching pairs
# Files will be matched by replacing PATTERN_GT with PATTERN_PRED
# Example: {file}_masks.tif will match with {file}_restored_masks.tif
PATTERN_GT="_masks.tif"
PATTERN_PRED="_restored_masks.tif"

# IoU threshold for object matching (0.0 to 1.0, typically 0.5)
IOU_THRESHOLD=0.5

# Output file for detailed results (optional, leave empty to skip saving)
RESULTS_FILE="${GT_DIR}/comparison_results.txt"

# Script directory
SCRIPT_DIR="/users/kir-fritzsche/aif490/devel/tissue_analysis/segmentation_scripts"

# =============================================================================
# RUN COMPARISON
# =============================================================================

echo "=============================================================================="
echo "Starting Segmentation Comparison"
echo "=============================================================================="
echo "Ground truth directory: ${GT_DIR}"
echo "Prediction directory: ${PRED_DIR}"
echo "GT pattern: ${PATTERN_GT}"
echo "Pred pattern: ${PATTERN_PRED}"
echo "IoU threshold: ${IOU_THRESHOLD}"
echo "Results file: ${RESULTS_FILE}"
echo ""

# Create Python script to run the comparison
python3 -u << EOF
import sys
sys.path.insert(0, '${SCRIPT_DIR}')

from compare_segmentations import batch_compare_segmentations

# Run batch comparison
results = batch_compare_segmentations(
    dir_gt='${GT_DIR}',
    dir_pred='${PRED_DIR}',
    pattern_gt='${PATTERN_GT}',
    pattern_pred='${PATTERN_PRED}',
    iou_threshold=${IOU_THRESHOLD},
    save_results='${RESULTS_FILE}' if '${RESULTS_FILE}' else None
)

if results is not None:
    print("\n" + "="*80)
    print("Comparison complete!")
    print("="*80)
else:
    print("\nERROR: Comparison failed or no matching pairs found")
    sys.exit(1)

EOF

echo ""
echo "=============================================================================="
echo "Comparison job finished"
echo "=============================================================================="
