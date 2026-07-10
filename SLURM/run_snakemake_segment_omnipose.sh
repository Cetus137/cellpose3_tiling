#!/bin/bash
# Wrapper script for omnipose 3D segmentation workflow

module load snakemake/8.4.2-foss-2023a

snakemake \
    --snakefile Snakefile_segment_omnipose \
    --executor slurm \
    --jobs 1200 \
    --default-resources slurm_partition=short slurm_account=kir.prj \
    --latency-wait 60 \
    "$@"
