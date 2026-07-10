#!/bin/bash
# Wrapper script for the cpSAM boundary-segmentation Snakemake workflow.
#
# Usage:
#   cd /users/kir-fritzsche/aif490/devel/tissue_analysis/segmentation_scripts/SLURM
#   bash run_snakemake_cpSAM_boundary.sh
#
# Dry-run (preview jobs without submitting):
#   bash run_snakemake_cpSAM_boundary.sh --dry-run
#
# All extra arguments are forwarded to snakemake via "$@".

module load snakemake/8.4.2-foss-2023a

snakemake \
    --snakefile Snakefile_cpSAM_boundary \
    --executor slurm \
    --jobs 3000 \
    --default-resources slurm_partition=short slurm_account=kir.prj mem_mb=0 \
    --latency-wait 60 \
    --restart-times 1 \
    "$@"
