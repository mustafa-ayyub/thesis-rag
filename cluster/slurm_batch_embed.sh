#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Batch embedding job — SLURM job for document ingestion (one-shot)
#
# Usage:
#   sbatch cluster/slurm_batch_embed.sh <input.json> <output.json>
#
# Example:
#   sbatch cluster/slurm_batch_embed.sh data/chunks_export.json data/vectors.json
#
# After the job completes, scp the output file back to your laptop:
#   scp -J u29661@glogin-gpu.hpc.gwdg.de \
#       u29661@<node>:~/thesis-rag/data/vectors.json \
#       data/vectors.json
# Then run scripts/ingest.py --vectors data/vectors.json to load into Qdrant.
# ──────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=thesis-batch-embed
#SBATCH --account=kisski-bingen-ma
#SBATCH --partition=kisski
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:A100:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/batch_embed_%j.out
#SBATCH --error=logs/batch_embed_%j.err

set -euo pipefail
mkdir -p logs

INPUT_FILE="${1:?Usage: sbatch slurm_batch_embed.sh <input.json> <output.json>}"
OUTPUT_FILE="${2:?Usage: sbatch slurm_batch_embed.sh <input.json> <output.json>}"

echo "Input:  ${INPUT_FILE}"
echo "Output: ${OUTPUT_FILE}"

source ~/thesis-venv-gpu/bin/activate

# Compute nodes have no internet — models are pre-cached in $HOME, load offline
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

REPO_PATH="${HOME}/thesis-rag"
cd "${REPO_PATH}"

python cluster/batch_embed.py \
    --input  "${INPUT_FILE}" \
    --output "${OUTPUT_FILE}"

echo "Done. Download with:"
echo "  scp -J u29661@glogin-gpu.hpc.gwdg.de u29661@$(hostname):${REPO_PATH}/${OUTPUT_FILE} ./"
