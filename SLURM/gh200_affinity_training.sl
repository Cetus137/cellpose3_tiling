#!/bin/bash
#SBATCH --job-name      gh200_affinity
#SBATCH --account       gpu_kir.prj
#SBATCH --partition     gpu_gh200_144gb
#SBATCH --gres          gpu:1
#SBATCH --cpus-per-task 8
#SBATCH --mem           64G
#SBATCH --time          4:00:00
#SBATCH --output        slogs/affinity/gh200_affinity_training.%j.out
#SBATCH --error         slogs/affinity/gh200_affinity_training.%j.err

# Run gh200_setup_env.sl once before using this script.
#
# To train from scratch:   leave RESUME empty.
# To resume from checkpoint: set RESUME to the checkpoint path, e.g.:
#   RESUME="$OUTPUT_DIR/affinity_unet_ep0050.pt"

SCRIPTS=/users/kir-fritzsche/aif490/devel/tissue_analysis/segmentation_scripts/scripts
TRAIN_DIR=/users/kir-fritzsche/aif490/devel/tissue_analysis/segmentation_scripts/for_training/ph3/training_boundary
OUTPUT_DIR=/users/kir-fritzsche/aif490/devel/tissue_analysis/segmentation_scripts/models/affinity_unet_ph3_centroid
VENV=/users/kir-fritzsche/aif490/devel/venv/gh200_affinity_env

#RESUME=""
RESUME="$OUTPUT_DIR/affinity_unet_ep0130.pt"

mkdir -p "$OUTPUT_DIR"
mkdir -p slogs/affinity

# ── aarch64 tooling ───────────────────────────────────────────────────────────
export PATH=/apps/kir/eb/hpc-utils/aarch64:$PATH
module purge
module use /apps/eb/el9/2025a/aarch64/modules/all
module load CUDA/12.6.0
source "$VENV/bin/activate"

echo "Node        : $(hostname)"
echo "Architecture: $(uname -m)"
echo "GPU         : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

# ── Resume flag ───────────────────────────────────────────────────────────────
RESUME_FLAG=""
if [ -n "$RESUME" ]; then
    RESUME_FLAG="--resume $RESUME"
fi

# ── Train ─────────────────────────────────────────────────────────────────────
python3 -u "$SCRIPTS/affinity_training.py" \
    --train_dir       "$TRAIN_DIR" \
    --output_dir      "$OUTPUT_DIR" \
    --epochs          200 \
    --batch_size      4 \
    --lr              2e-4 \
    --num_workers     8 \
    --save_every      5 \
    --vis_every       5 \
    --lambda_centroid 0.1 \
    --centroid_sigma  3.0 \
    $RESUME_FLAG \
    --gpu
