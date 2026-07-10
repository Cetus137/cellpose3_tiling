#!/bin/bash
#SBATCH --job-name      gh200_setup
#SBATCH --account       gpu_kir.prj
#SBATCH --partition     gpu_gh200_144gb
#SBATCH --gres          gpu:1
#SBATCH --cpus-per-task 8
#SBATCH --mem           32G
#SBATCH --time          00:30:00
#SBATCH --output        slogs/affinity/gh200_setup.%j.out
#SBATCH --error         slogs/affinity/gh200_setup.%j.err

# One-time setup: builds an aarch64-native Python venv for affinity training
# on the GH200 nodes.  Run once with:
#   sbatch SLURM/gh200_setup_env.sl
# Then use SLURM/gh200_affinity_training.sl for all subsequent training jobs.

set -e

VENV=/users/kir-fritzsche/aif490/devel/venv/gh200_affinity_env

mkdir -p slogs/affinity

# ── Redirect uv cache to scratch to avoid home-directory quota ────────────────
export UV_CACHE_DIR=/well/kir-fritzsche/users/aif490/tmp/uv_cache
mkdir -p "$UV_CACHE_DIR"

# ── aarch64 tooling ───────────────────────────────────────────────────────────
export PATH=/apps/kir/eb/hpc-utils/aarch64:$PATH
module purge
module use /apps/eb/el9/2025a/aarch64/modules/all
module load CUDA/12.6.0

echo "Architecture : $(uname -m)"
echo "uv           : $(uv --version)"
# nvcc in the standard module tree is x86_64 — skip that check
echo "CUDA libs    : $(ldconfig -p 2>/dev/null | grep libcuda | head -1 || echo 'loaded via module')"

# ── Create venv ───────────────────────────────────────────────────────────────
echo "Creating venv at $VENV ..."
uv venv "$VENV" --python 3.11 --clear
source "$VENV/bin/activate"

# ── PyTorch (CUDA 12.6, includes sm_90a kernels for GH200) ───────────────────
echo "Installing PyTorch ..."
uv pip install --no-cache torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu126

# ── Scientific stack ──────────────────────────────────────────────────────────
echo "Installing scientific dependencies ..."
uv pip install --no-cache \
    tifffile \
    scikit-image \
    scipy \
    numba \
    natsort \
    matplotlib \
    pandas \
    numpy

# ── Verify ────────────────────────────────────────────────────────────────────
python3 - <<'EOF'
import torch
print(f"PyTorch  : {torch.__version__}")
print(f"CUDA available : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU      : {torch.cuda.get_device_name(0)}")
    print(f"VRAM     : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
import tifffile, skimage, scipy, numba, natsort, matplotlib
print("All packages imported successfully.")
EOF

echo "Setup complete. Venv: $VENV"
