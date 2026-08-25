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

## Development branches

| Branch | Scope |
|---|---|
| `main` | stable releases only |
| `month-2/shared-retrieval-pipeline` | Month 2 — shared ingestion + retrieval pipeline |
| `month-3/system-a-b` | Month 3 — System A and System B |
| `month-4/evaluation` | Month 4 — RAGAS evaluation and analysis |
