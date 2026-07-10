#!/bin/bash
#SBATCH --job-name      temporal_affinity_train
#SBATCH --account       gpu_kir.prj
#SBATCH --partition=gpu_a100_40gb,gpu_rtx8000_48gb,gpu_v100_32gb,gpu_a100_80gb
#SBATCH --time=8:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-gpu=40G
#SBATCH --exclude       compg009,compg010,compg011,compg013
#SBATCH --output        slogs/affinity/temporal_affinity_training.%j.out
#SBATCH --error         slogs/affinity/temporal_affinity_training.%j.err

module purge
source /well/kir/config/modules.sh
module load Python/3.10.8-GCCcore-12.2.0
source ~/devel/venv/Python-3.10.8-GCCcore-12.2.0/cellpose3_env/bin/activate

SCRIPTS=/users/kir-fritzsche/aif490/devel/tissue_analysis/segmentation_scripts/scripts
MODELS=/users/kir-fritzsche/aif490/devel/tissue_analysis/segmentation_scripts/models

# ── Branch (b): real-frame triplets from extract_temporal_triplets.py ─────────
TRAIN_DIR=/users/kir-fritzsche/aif490/devel/tissue_analysis/segmentation_scripts/for_training/temporal/pos2_crop1
# ── Branch (a): static annotated tiles for synthetic-motion alignment (optional).
#    Leave empty to train on branch (b) only.
#SYNTHETIC_DIR=/users/kir-fritzsche/aif490/devel/tissue_analysis/segmentation_scripts/for_training/ph3/training_boundary

OUTPUT_DIR=$MODELS/affinity_unet_ph3_temporal

# Warm-start (1->3 channels) from the frozen per-frame model, OR resume a
# temporal checkpoint. Exactly one of these should be set.
WARM_START="$MODELS/affinity_unet_ph3_centroid/affinity_unet_ep0200.pt"
RESUME=""

mkdir -p "$OUTPUT_DIR" slogs/affinity

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

python3 -u "$SCRIPTS/temporal_affinity_training.py" \
    --train_dir         "$TRAIN_DIR" \
    $SYNTH_FLAG \
    --output_dir        "$OUTPUT_DIR" \
    $INIT_FLAG \
    --epochs            80 \
    --batch_size        2 \
    --lr                1e-4 \
    --num_workers       4 \
    --save_every        5 \
    --lambda_fg         0.1 \
    --lambda_centroid   0.1 \
    --centroid_sigma    3.0 \
    --target_synth_frac 0.7 \
    --branch_b_warmup   15 \
    --motion_end        15.0 \
    --motion_ramp       40 \
    --gpu
