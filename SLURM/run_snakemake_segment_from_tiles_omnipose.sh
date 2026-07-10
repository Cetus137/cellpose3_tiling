#!/bin/bash
# Wrapper script for omnipose from-tiles segmentation workflow

module load snakemake/8.4.2-foss-2023a

snakemake \
    --snakefile Snakefile_segment_from_tiles_omnipose \
    --executor slurm \
    --jobs 3000 \
    --default-resources slurm_partition=short slurm_account=kir.prj \
    --restart-times 1 \
    --latency-wait 60 \
    "$@"
