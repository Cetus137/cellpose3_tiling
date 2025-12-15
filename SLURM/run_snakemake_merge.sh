#!/bin/bash
# cluster_submit.sh - Wrapper script for Snakemake SLURM submission

# Load Snakemake module
module load snakemake/8.4.2-foss-2023a

snakemake \
    --snakefile Snakefile_merge \
    --executor slurm \
    --jobs 65 \
    --default-resources slurm_partition=short slurm_account=kir.prj \
    --latency-wait 60 \
    "$@"
