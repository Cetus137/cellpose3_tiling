#!/bin/bash
#SBATCH --job-name      extract_temporal_data
#SBATCH --account       kir.prj
#SBATCH --partition     short
#SBATCH --cpus-per-task 4
#SBATCH --mem           64G
#SBATCH --time          03:00:00
#SBATCH --exclude       compg009,compg010,compg011,compg013
#SBATCH --output        slogs/temporal/extract_temporal_data.%j.out
#SBATCH --error         slogs/temporal/extract_temporal_data.%j.err

# ---------------------------------------------------------------------------
# Build the temporal-boundary-model training set in two chained steps:
#   1. select_temporal_anchors.py   (SOLVED PG DB  -> anchors.csv)
#   2. extract_temporal_triplets.py (zarrs+anchors -> {stem}_stack/_masks/_weight.tif)
# Both run on CPU in ultrack_env. The PG server must be UP for step 1
# (see lymphnode_analysis/SLURM/pg_server.address).
# ---------------------------------------------------------------------------

module purge
source /well/kir/config/modules.sh
module load Python/3.10.8-GCCcore-12.2.0
source ~/devel/venv/Python-3.10.8-GCCcore-12.2.0/ultrack_env/bin/activate

mkdir -p slogs/temporal

SCRIPTS=/users/kir-fritzsche/aif490/devel/tissue_analysis/segmentation_scripts/scripts

# ---------------------------------------------------------------------------
# Inputs — edit before submitting
# ---------------------------------------------------------------------------

# Solved ultrack PG DB. Defaults to the address written by the running PG server;
# override with an explicit "user:pass@host:port/db?..." string if needed.
PG_ADDR_FILE=/users/kir-fritzsche/aif490/devel/tissue_analysis/lymphnode_analysis/SLURM/pg_server.address
PG_ADDRESS="$(cat "$PG_ADDR_FILE" 2>/dev/null)"

# Image source (same voxel frame as the DB node coords).
#   IMAGE_ZARR : (T,Z,Y,X) timelapse -> the intensity frames of {stem}_stack.tif.
#                raw_data.zarr is RAW domain — the model infers on RESTORED tiles, so point
#                this at a restored volume for a domain-matched set once one exists (TODO).
# Center-frame MASKS are read from the solved DB (nodes.pickle), NOT a seg zarr — this
# guarantees they match the exact solve the anchors came from. No SEG_ZARR needed.
IMAGE_ZARR=/users/kir-fritzsche/aif490/devel/tissue_analysis/lymphnode_analysis/data2track/b2-6a-pos2-01_deskew_cgt/crop1/raw_data.zarr

# Segmentation volume shape (Z Y X) — for the FOV-border margin in step 1.
IMAGE_SHAPE="256 1294 1228"

# A short tag identifying this crop; used for the output stems + sub-dir.
TAG=pos2_crop1

OUT_ROOT=/users/kir-fritzsche/aif490/devel/tissue_analysis/segmentation_scripts/for_training/temporal
OUT_DIR="$OUT_ROOT/$TAG"
ANCHORS_CSV="$OUT_DIR/anchors.csv"

# Temporal half-window: K = 2*HALF+1. Use 2 (K=5) to enable Noise2Noise
# neighbour-supervision in TemporalAffinityDataset; 1 (K=3, triplet) is direct-only.
HALF_WINDOW=2

# Which anchor categories to extract to disk (jittery_hard is validation-only).
# Start with trusted_accurate only — the clean N2N training core (~1221 anchors, ~38 GB at
# uint16). Add "trusted_dynamic" (the over-smoothing guard, ~4451 more, ~176 GB total) in a
# second extract once disk allows / the model trains.
CATEGORIES="trusted_accurate"

mkdir -p "$OUT_DIR"

echo "===== extract_temporal_data ====="
echo "  PG address   : ${PG_ADDRESS:-<EMPTY — server not up?>}"
echo "  image zarr   : $IMAGE_ZARR"
echo "  masks source : DB nodes.pickle (no seg zarr)"
echo "  image shape  : $IMAGE_SHAPE"
echo "  out dir      : $OUT_DIR"
echo "  half window  : $HALF_WINDOW  (K=$((2*HALF_WINDOW+1)))"
echo "  categories   : $CATEGORIES"
echo "=================================="

if [ -z "$PG_ADDRESS" ]; then
    echo "ERROR: no PG address (is the server up? check $PG_ADDR_FILE)"; exit 1
fi

# ---------------------------------------------------------------------------
# Step 1 — select anchors from the solved DB
# ---------------------------------------------------------------------------
echo "[1/2] selecting temporal anchors ..."
python3 -u "$SCRIPTS/select_temporal_anchors.py" \
    --pg_address  "$PG_ADDRESS" \
    --image_shape  $IMAGE_SHAPE \
    --out_csv     "$ANCHORS_CSV" || { echo "step 1 failed"; exit 1; }

if [ ! -s "$ANCHORS_CSV" ]; then
    echo "ERROR: no anchors written — loosen thresholds in select_temporal_anchors.py"; exit 1
fi

# ---------------------------------------------------------------------------
# Step 2 — extract co-located temporal triplets
# ---------------------------------------------------------------------------
echo "[2/2] extracting temporal triplets ..."
python3 -u "$SCRIPTS/extract_temporal_triplets.py" \
    --manifest    "$ANCHORS_CSV" \
    --pg_address  "$PG_ADDRESS" \
    --image_zarr  "$IMAGE_ZARR" \
    --out_dir     "$OUT_DIR" \
    --crop        128 128 128 \
    --half_window "$HALF_WINDOW" \
    --categories  $CATEGORIES \
    --tag         "$TAG" || { echo "step 2 failed"; exit 1; }

echo "extract_temporal_data complete -> $OUT_DIR"
