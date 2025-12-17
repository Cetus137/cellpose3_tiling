#!/bin/bash
# Wrapper script for Snakemake segmentation workflow

# Load Snakemake module
module load snakemake/8.4.2-foss-2023a

# Set temporary directory to workspace (avoid full /tmp)
export TMPDIR=/well/kir-fritzsche/users/aif490/tmp
mkdir -p $TMPDIR

snakemake \
    --snakefile Snakefile_segment_independent_views \
    --executor slurm \
    --jobs 10 \
    --default-resources slurm_partition=short slurm_account=kir.prj \
    --latency-wait 60 \
    --restart-times 3 \
    "$@"
