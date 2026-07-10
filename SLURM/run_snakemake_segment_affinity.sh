#!/bin/bash
# Run the affinity segmentation Snakemake workflow.
# Submit from: SLURM/

module load snakemake/8.4.2-foss-2023a

snakemake \
    --snakefile Snakefile_segment_affinity \
    --executor slurm \
    --jobs 2000 \
    --default-resources slurm_partition=short slurm_account=kir.prj \
    --latency-wait 60 \
    --restart-times 1 \
    "$@"
