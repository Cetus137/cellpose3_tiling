#!/bin/bash
# Wrapper script for Snakemake segmentation workflow

# Load Snakemake module
module load snakemake/8.4.2-foss-2023a

snakemake \
    --snakefile Snakefile_segment_from_tiles \
    --executor slurm \
    --jobs 1000 \
    --default-resources slurm_partition=short slurm_account=kir.prj \
    --restart-times 0 \
    --latency-wait 60 \
    "$@"
