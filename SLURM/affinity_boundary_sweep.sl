#!/bin/bash
#SBATCH --job-name      aff_boundary_sweep
#SBATCH --cpus-per-task 2
#SBATCH --mem           64G
#SBATCH --time          01:00:00
#SBATCH --output        slogs/aff_boundary_sweep.%j.out
#SBATCH --error         slogs/aff_boundary_sweep.%j.err
#SBATCH --exclude       compg009,compg010,compg011,compg013

# CPU-only: collapses the reconstructed 9-channel affinities into boundary maps
# under several reduction methods and writes a montage PNG for visual comparison.

module purge
source /well/kir/config/modules.sh
module load Python/3.10.8-GCCcore-12.2.0
source ~/devel/venv/Python-3.10.8-GCCcore-12.2.0/cellpose3_env/bin/activate

SCRIPTS=/users/kir-fritzsche/aif490/devel/tissue_analysis/segmentation_scripts/scripts
AFF_DIR=/users/kir-fritzsche/aif490/devel/tissue_analysis/lymphnode_analysis/data2track/b2-6a_overview_pos1-01_deskew_cgt/crop1/tiles_restored_affinity_raw/reconstructed

mkdir -p slogs

python3 -u "$SCRIPTS/affinity_boundary_sweep.py" \
    --aff_dir    "$AFF_DIR" \
    --out_dir    "$AFF_DIR/boundary_sweep" \
    --timepoint  0 \
    --mask_fg
# add --no_tif to write only the montage PNG (skip the 7 per-method volumes)
