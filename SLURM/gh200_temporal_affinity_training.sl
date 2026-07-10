#!/bin/bash
#SBATCH --job-name      gh200_temporal_affinity
#SBATCH --account       gpu_kir.prj
#SBATCH --partition     gpu_gh200_144gb
#SBATCH --gres          gpu:1
#SBATCH --cpus-per-task 8
#SBATCH --mem           64G
#SBATCH --time          8:00:00
#SBATCH --output        slogs/affinity/gh200_temporal_affinity_training.%j.out
#SBATCH --error         slogs/affinity/gh200_temporal_affinity_training.%j.err

# gh200 (aarch64) version of temporal_affinity_training.sl.
# Run gh200_setup_env.sl once before using this script.
#
# Warm-start (1->3 channels) from the frozen per-frame model, OR resume a
# temporal checkpoint. Exactly one of WARM_START / RESUME should be set.

SCRIPTS=/users/kir-fritzsche/aif490/devel/tissue_analysis/segmentation_scripts/scripts
MODELS=/users/kir-fritzsche/aif490/devel/tissue_analysis/segmentation_scripts/models
VENV=/users/kir-fritzsche/aif490/devel/venv/gh200_affinity_env

# ── Branch (b): real-frame triplets from extract_temporal_triplets.py ─────────
TRAIN_DIR=/users/kir-fritzsche/aif490/devel/tissue_analysis/segmentation_scripts/for_training/temporal/pos2_crop1
# ── Branch (a): static annotated tiles for synthetic-motion alignment (optional).
#    Uncomment to enable; leave unset to train on branch (b) only.
#SYNTHETIC_DIR=/users/kir-fritzsche/aif490/devel/tissue_analysis/segmentation_scripts/for_training/ph3/training_boundary

OUTPUT_DIR=$MODELS/affinity_unet_ph3_temporal

WARM_START="$MODELS/affinity_unet_ph3_centroid/affinity_unet_ep0200.pt"
RESUME=""

mkdir -p "$OUTPUT_DIR" slogs/affinity

# ── aarch64 tooling ───────────────────────────────────────────────────────────
export PATH=/apps/kir/eb/hpc-utils/aarch64:$PATH
module purge
module use /apps/eb/el9/2025a/aarch64/modules/all
module load CUDA/12.6.0
source "$VENV/bin/activate"

echo "Node        : $(hostname)"
echo "Architecture: $(uname -m)"
echo "GPU         : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

# ── Init flag: resume (strict, in_channels=3) XOR warm-start (1->3 channels) ──
INIT_FLAG=""
if [ -n "$RESUME" ]; then
    INIT_FLAG="--resume $RESUME"
elif [ -n "$WARM_START" ]; then
    INIT_FLAG="--warm_start $WARM_START"
fi

SYNTH_FLAG=""
if [ -n "$SYNTHETIC_DIR" ]; then
    SYNTH_FLAG="--synthetic_dir $SYNTHETIC_DIR"
fi

# ── Train ─────────────────────────────────────────────────────────────────────
python3 -u "$SCRIPTS/temporal_affinity_training.py" \
    --train_dir         "$TRAIN_DIR" \
    $SYNTH_FLAG \
    --output_dir        "$OUTPUT_DIR" \
    $INIT_FLAG \
    --epochs            80 \
    --batch_size        4 \
    --lr                1e-4 \
    --num_workers       8 \
    --save_every        5 \
    --lambda_fg         0.1 \
    --lambda_centroid   0.1 \
    --centroid_sigma    3.0 \
    --target_synth_frac 0.7 \
    --branch_b_warmup   15 \
    --motion_end        15.0 \
    --motion_ramp       40 \
    --gpu
