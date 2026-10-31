#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Interactive GPU server — SLURM job for live embed/rerank via SSH tunnel
#
# Submit: sbatch cluster/slurm_interactive.sh
# Then:   tail -f logs/gpu_server_<JOBID>.out   (to get the compute node name)
# Then:   ./scripts/tunnel.sh <compute-node>     (on your laptop)
# Then:   set gpu.remote_url: "http://localhost:8000" in config/settings.yaml
# ──────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=thesis-gpu-server
#SBATCH --account=kisski-bingen-ma
#SBATCH --partition=grete:interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:2g.20gb:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/gpu_server_%j.out
#SBATCH --error=logs/gpu_server_%j.err

set -euo pipefail
mkdir -p logs

NODE=$(hostname)
echo "==================================================================="
echo "  GPU server starting on node: ${NODE}"
echo "  Job ID: ${SLURM_JOB_ID}"
echo ""
echo "  Create SSH tunnel on your laptop:"
echo "    ./scripts/tunnel.sh ${NODE}"
echo ""
echo "  Then set in config/settings.yaml:"
echo "    gpu:"
echo "      remote_url: \"http://localhost:8000\""
echo "==================================================================="

source ~/thesis-venv-gpu/bin/activate

# Compute nodes have no internet — models are pre-cached in $HOME, load offline
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Adjust REPO_PATH if the repo is cloned to a different location on the cluster
REPO_PATH="${HOME}/thesis-rag"
cd "${REPO_PATH}"

uvicorn cluster.gpu_server:app \
    --host 0.0.0.0 \
    --port 8000 \
    --log-level info
