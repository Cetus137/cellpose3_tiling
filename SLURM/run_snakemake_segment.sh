#!/bin/bash
# Wrapper script for Snakemake segmentation workflow

# Load Snakemake module
module load snakemake/8.4.2-foss-2023a

snakemake \
    --snakefile Snakefile_segment \
    --executor slurm \
    --jobs 65 \
    --default-resources slurm_partition=short slurm_account=kir.prj \
    --latency-wait 240 \
    "$@"
