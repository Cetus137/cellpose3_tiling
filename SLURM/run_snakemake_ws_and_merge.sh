#!/bin/bash
# Wrapper script for Snakemake watershed + merge workflow

# Load Snakemake module
module load snakemake/8.4.2-foss-2023a

snakemake \
    --snakefile Snakefile_ws_and_merge \
    --executor slurm \
    --jobs 180 \
    --default-resources slurm_partition=short slurm_account=kir.prj \
    --latency-wait 240 \
    "$@"
