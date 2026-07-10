#!/bin/bash
# Wrapper script for Snakemake segmentation workflow

# Load Snakemake module
module load snakemake/8.4.2-foss-2023a

snakemake \
    --snakefile Snakefile_segment_cp4 \
    --executor slurm \
    --jobs 1200 \
    --default-resources slurm_partition=short slurm_account=kir.prj mem_mb=0 \
    --latency-wait 60 \
    "$@"
