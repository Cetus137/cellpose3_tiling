#!/bin/bash

#SBATCH --job-name      compare_seg
#SBATCH --cpus-per-task 1
#SBATCH --mem           4G
#SBATCH --time          03:00:00
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

# All config below can be overridden at submit time via:
#   sbatch --export=ALL,PRED_DIR=...,RESULTS_FILE=... compare_segmentations.sl
# The defaults reproduce the ph_segmented vs lifeact_seg_cp3 comparison.
TILES_DIR="/users/kir-fritzsche/aif490/devel/tissue_analysis/segmentation_scripts/for_training/ph3/tiles"
RESULTS_DIR="/users/kir-fritzsche/aif490/devel/tissue_analysis/segmentation_scripts/for_training/ph3/comparison_results"

# Directories containing segmentations to compare
PRED_DIR="${PRED_DIR:-${TILES_DIR}/lifeact_seg_cp3}"
GT_DIR="${GT_DIR:-${TILES_DIR}/ph_segmented}"

# File patterns for matching pairs
# For position-based matching: use glob patterns (e.g., *_masks.tif, *.tif) - wildcards OK
# For similarity matching: use glob patterns to find files, then match by string similarity
# For basename matching: use literal suffixes (e.g., _masks.tif) - NO wildcards
# NOTE: With position matching, both directories should use the same glob pattern to find corresponding files
PATTERN_GT="${PATTERN_GT:-*.tif}"
PATTERN_PRED="${PATTERN_PRED:-*.tif}"

# Matching strategy: "position", "similarity", or "basename"
# - "position": Sort and match by index (simle, but fails if files are interleaved)
# - "similarity": Find best string match for each file (BEST for mixed file groups like ordered/disordered)
# - "basename": Match by replacing pattern_gt with pattern_pred in filename
MATCH_BY="${MATCH_BY:-similarity}"

# IoU threshold for object matching (0.0 to 1.0, typically 0.5)
IOU_THRESHOLD="${IOU_THRESHOLD:-0.50}"

# Depth filter (pixels): only cells whose centroid along the depth axis (z, axis 0)
# is < MAX_DEPTH are included in the comparison. Tiles are 256 px deep in z, so
# MAX_DEPTH=128 keeps the shallow half. Set to empty ("") to disable filtering.
MAX_DEPTH="${MAX_DEPTH:-128}"
DEPTH_AXIS="${DEPTH_AXIS:-0}"

# Output file for detailed results (optional, leave empty to skip saving)
# A depth-filtered run writes to a separate file so full-depth results are preserved.
RESULTS_FILE="${RESULTS_FILE:-${RESULTS_DIR}/ph_lifeact_cp3_comparison_results_IoU_050_DICE_depth${MAX_DEPTH}.txt}"

# Script directory
SCRIPT_DIR="/users/kir-fritzsche/aif490/devel/tissue_analysis/segmentation_scripts"

# =============================================================================sb
# RUN COMPARISON
# =============================================================================

echo "=============================================================================="
echo "Starting Segmentation Comparison"
echo "=============================================================================="
echo "Ground truth directory: ${GT_DIR}"
echo "Prediction directory: ${PRED_DIR}"
echo "GT pattern: ${PATTERN_GT}"
echo "Pred pattern: ${PATTERN_PRED}"
echo "Matching strategy: ${MATCH_BY}"
echo "IoU threshold: ${IOU_THRESHOLD}"
echo "Max depth: ${MAX_DEPTH:-none} (axis ${DEPTH_AXIS})"
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
    save_results='${RESULTS_FILE}' if '${RESULTS_FILE}' else None,
    match_by='${MATCH_BY}',
    max_depth=${MAX_DEPTH:-None},
    depth_axis=${DEPTH_AXIS}
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
