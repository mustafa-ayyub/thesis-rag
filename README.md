# Thesis RAG

**RAG Pipelines as MCP Tools: A Comparative Study of Retrieval Accuracy, Response Quality, and Latency in Document-Heavy Workflows**

Master's thesis — TH Bingen, supervised by Prof. Simon Duque Antón.
Timeline: July 2026 – January 2027.

## Overview

Controlled comparison of two RAG architectures sharing the same hybrid retrieval pipeline:

| System | Description |
|---|---|
| **System A** | Standard RAG — fixed single-pass retrieval, one LLM call |
| **System B** | MCP-Wrapped RAG — ReAct agent, self-correcting retrieval via MCP tool |

Document corpus: ECB and Deutsche Bundesbank Financial Stability Reviews (~10–50 PDFs).

## Setup

```bash
# Create virtual environment
uv venv --python 3.14 .venv
source .venv/bin/activate

# Install dependencies
uv pip install --python .venv/bin/python -r requirements-dev.txt

# Configure
cp .env.example .env
# Edit .env and set SAIA_API_KEY

# Start Qdrant (Docker)
docker run -d -p 6333:6333 qdrant/qdrant
```

## Project structure

```
src/ingestion/   — PDF parsing, chunking, Qdrant indexing
src/retrieval/   — embedding, hybrid search, reranking, query rewriting
cluster/         — GPU server (FastAPI) and SLURM batch scripts for GWDG
scripts/         — ingest and validation entry points
tests/           — pytest suite
docs/            — design documents (chunking strategy, code standards)
config/          — settings.yaml template
```

## GPU compute (GWDG / KISSKI)

BGE-M3 and BGE-Reranker-v2-m3 run on GWDG's Grete GPU cluster (project `kisski-bingen-ma`).
See `cluster/` for SLURM scripts and `scripts/tunnel.sh` for the SSH tunnel helper.
The `gpu.remote_url` setting in `config/settings.yaml` switches between local and remote GPU mode.

## Running on another machine (Docker)

Works on Linux/macOS/Windows. The image contains **only the orchestration code** —
no torch, no model weights: embedding/reranking always runs on a GPU server
(see `cluster/`), reached via `GPU_REMOTE_URL`.

Prerequisites: Docker, a clone of this repo, a running GPU server
(see `cluster/README.md`), and network access from this machine to it
(directly, or via `scripts/tunnel.sh` on this machine).

```bash
# 1. configure
cp .env.example .env
#    SAIA_API_KEY=...
#    GPU_REMOTE_URL=http://host.docker.internal:8000
#    (host.docker.internal = this machine — use it when an SSH tunnel to the
#     GPU server is bound on this machine's port 8000; use
#     http://<gpu-host>:8000 when the server is directly reachable)

# 2. start Qdrant + build the app image
docker compose up -d qdrant
docker compose build app

# 3. first time on this machine only: copy data/ (documents/,
#    chunks_export.json, vectors.json, queries/, results/) from the original
#    machine, then load the index WITHOUT re-embedding:
docker compose run --rm app python scripts/ingest.py --vectors-file data/vectors.json

# 4. use it
docker compose run --rm app python scripts/validate_retrieval.py
```

Notes:
- Qdrant data lives in the `qdrant-data` Docker volume — deleting the volume
  wipes the index (re-run step 3).
- Re-embedding from scratch (no `vectors.json`) is done with a GPU batch job on
  a GPU machine — see `cluster/README.md`.

## Development branches

| Branch | Scope |
|---|---|
| `main` | stable releases only |
| `month-2/shared-retrieval-pipeline` | Month 2 — shared ingestion + retrieval pipeline |
| `month-3/system-a-b` | Month 3 — System A and System B |
| `month-4/evaluation` | Month 4 — RAGAS evaluation and analysis |
