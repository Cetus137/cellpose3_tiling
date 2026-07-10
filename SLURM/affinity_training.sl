#!/bin/bash
#SBATCH --job-name      affinity_train
#SBATCH --account       gpu_kir.prj
#SBATCH --partition=gpu_a100_40gb,gpu_rtx8000_48gb,gpu_v100_32gb,gpu_a100_80gb
#SBATCH --time=4:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-gpu=40G
#SBATCH --exclude       compg009,compg010,compg011,compg013
#SBATCH --output        slogs/affinity/affinity_training.%j.out
#SBATCH --error         slogs/affinity/affinity_training.%j.err

module purge
source /well/kir/config/modules.sh
module load Python/3.10.8-GCCcore-12.2.0
source ~/devel/venv/Python-3.10.8-GCCcore-12.2.0/cellpose3_env/bin/activate

SCRIPTS=/users/kir-fritzsche/aif490/devel/tissue_analysis/segmentation_scripts/scripts
TRAIN_DIR=/users/kir-fritzsche/aif490/devel/tissue_analysis/segmentation_scripts/for_training/ph3/training_boundary
OUTPUT_DIR=/users/kir-fritzsche/aif490/devel/tissue_analysis/segmentation_scripts/models/affinity_unet_ph3_centroid

# Set to the checkpoint path to resume, or leave empty to train from scratch.
#RESUME=""
RESUME="$OUTPUT_DIR/affinity_unet_ep0130.pt"

mkdir -p "$OUTPUT_DIR"
mkdir -p slogs/affinity

RESUME_FLAG=""
if [ -n "$RESUME" ]; then
    RESUME_FLAG="--resume $RESUME"
fi

python3 -u "$SCRIPTS/affinity_training.py" \
    --train_dir       "$TRAIN_DIR" \
    --output_dir      "$OUTPUT_DIR" \
    --epochs          200 \
    --batch_size      2 \
    --lr              2e-4 \
    --num_workers     4 \
    --save_every      5 \
    --vis_every       5 \
    --lambda_centroid 0.1 \
    --centroid_sigma  3.0 \
    $RESUME_FLAG \
    --gpu
