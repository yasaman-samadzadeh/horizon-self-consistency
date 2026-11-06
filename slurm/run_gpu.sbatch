#!/bin/bash
# Run one module of the package on one GPU, from the repository root (after `uv sync` and
# `python scripts/download_models.py`; create logs/ first: `mkdir -p logs`).
#   sbatch --job-name=hsc-main slurm/run_gpu.sbatch hsc.audit.run_tsfm --models chronos-bolt-small --device cuda
# Replace the placeholders with your cluster's values. We used one NVIDIA A2 (16 GB) per job: --gres=gpu:a2:1.
#SBATCH --partition=<PARTITION>
#SBATCH --qos=<QOS>
#SBATCH --account=<ACCOUNT>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
echo "job=${SLURM_JOB_ID:-local} gpu=${CUDA_VISIBLE_DEVICES:-none} start=$(date -u +%FT%TZ)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
.venv/bin/python -m "$@"
echo "end=$(date -u +%FT%TZ)"
