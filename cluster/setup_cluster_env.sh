#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# One-time cluster environment setup — run this on GWDG the first time
#
# ssh u29661@glogin-gpu.hpc.gwdg.de
# git clone <your-repo-url> ~/thesis-rag
# bash ~/thesis-rag/cluster/setup_cluster_env.sh
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

VENV_PATH="${HOME}/thesis-venv-gpu"
REPO_PATH="${HOME}/thesis-rag"

# Check available Python modules with: module avail python
# Adjust the module name to match what GWDG offers
module load python/3.11 2>/dev/null || module load python 2>/dev/null || true

python3 -m venv "${VENV_PATH}"
source "${VENV_PATH}/bin/activate"
pip install --upgrade pip

# Install PyTorch for the cluster CUDA version.
# Check with: nvcc --version  — Grete typically runs CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121

pip install -r "${REPO_PATH}/requirements-gpu.txt"

echo ""
echo "Cluster environment ready at ${VENV_PATH}"
echo "Activate with: source ${VENV_PATH}/bin/activate"
