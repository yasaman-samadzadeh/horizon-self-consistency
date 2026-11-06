#!/bin/bash
# Run one module of the package on CPUs only (baselines, regime switching, residual-noise baseline).
#   sbatch --job-name=hsc-rigour-epf slurm/run_cpu.sbatch hsc.audit.rigour floors --sources epf --workers 16
# Replace the placeholders with your cluster's values; create logs/ first (`mkdir -p logs`).
#SBATCH --partition=<PARTITION>
#SBATCH --qos=<QOS>
#SBATCH --account=<ACCOUNT>
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
echo "job=${SLURM_JOB_ID:-local} cpus=${SLURM_CPUS_PER_TASK:-?} start=$(date -u +%FT%TZ)"
.venv/bin/python -m "$@"
echo "end=$(date -u +%FT%TZ)"
