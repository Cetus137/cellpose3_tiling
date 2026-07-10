#!/bin/bash
# Wrapper script for seeded omnipose 3D segmentation workflow.
#
# Usage:
#   cd /users/kir-fritzsche/aif490/devel/tissue_analysis/segmentation_scripts/SLURM
#   bash run_snakemake_segment_omnipose_seeded.sh
#
# Dry-run (preview jobs without submitting):
#   bash run_snakemake_segment_omnipose_seeded.sh --dry-run

module load snakemake/8.4.2-foss-2023a

snakemake \
    --snakefile Snakefile_segment_omnipose_seeded \
    --executor slurm \
    --jobs 3000 \
    --default-resources slurm_partition=short slurm_account=kir.prj mem_mb=0 \
    --latency-wait 60 \
    --restart-times 1 \
    "$@"
