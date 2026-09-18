# Thesis RAG

**RAG Pipelines as MCP Tools: A Comparative Study of Retrieval Accuracy, Response Quality, and Latency in Document-Heavy Workflows**

Master's thesis — TH Bingen, supervised by Prof. Simon Duque Antón.
Timeline: July 2026 – January 2027.

## Overview

Controlled comparison of two RAG architectures sharing an identical retrieval
pipeline — only the control layer differs:

| System | Description | Status |
|---|---|---|
| **System A** | Standard RAG — fixed single-pass retrieval, one LLM call | in progress (Month 3) |
| **System B** | MCP-Wrapped RAG — ReAct agent, self-correcting retrieval via MCP tool | planned (Month 4) |

**Research question:** How does wrapping a RAG pipeline as an MCP tool affect
retrieval accuracy, response quality, and latency?

**Corpus:** European Central Bank (ECB) and Deutsche Bundesbank Financial
Stability Reviews. Currently 5 editions (665 pages): Bundesbank FSR 2023/2024/
2025, ECB FSR 2025/2026. The corpus will grow to ~10–50 PDFs (400–600 pages)
before the final evaluation. Documents contain prose, tables, charts, and
images.

**Benchmark:** 95 human-annotated queries (`data/queries/queries.json`) with
ground-truth answers, supporting contexts, and source (document, page) pairs:

| Type | Count | Role |
|---|---|---|
| `simple_factual` | 26 | single-step retrieval |
| `multi_step` | 39 | answers spanning multiple passages/pages |
| `edge_cases` | 15 | ambiguous, cross-edition, and other hard cases |
| `no_retrieval` | 15 | out-of-corpus; excluded from retrieval metrics, used to test refusal |

## Retrieval pipeline

```
PDF ─► parser        pdfplumber tables + pymupdf text; headers/footers/
                     footnotes stripped; chart plot areas rejected (they are
                     mis-detected as tables by pdfplumber)
    ─► chunker       recursive text chunks (512 tokens max / 64 overlap,
                     cl100k_base); each real table = one atomic markdown chunk
    ─► indexer       BGE-M3 dense + sparse vectors (one forward pass) →
                     multi-vector Qdrant points
```

```
query ─► rewriter    SAIA qwen3.8-27b (thinking off) rewrites the question
                     into a keyword-rich search query
    ─► embedder      BGE-M3 → (dense 1024-d, sparse)
    ─► hybrid search dense + sparse Qdrant queries merged via Reciprocal
                     Rank Fusion (k=60) → top 20
    ─► reranker      BGE-reranker-v2-m3 cross-encoder → top 5
                     (top 10 during validation)
```

The retriever interface is frozen for the A/B comparison:
`Retriever.retrieve(query: str) -> list[RetrievedChunk]`. System B wraps this
exact pipeline as an MCP tool.

## Benchmark metrics

Retrieval quality is measured per query type on the 80 scored queries:

- **P@5** — normalized precision: fraction of the query's relevant
  (document, page) pairs found in the top 5 (a full hit scores 1.0)
- **R@10**, **nDCG@10**, **MRR**

Matching is on **(document, page)** pairs, not page alone — the corpus PDFs
share page numbers, so a page number is only meaningful together with its
document.

**Current baseline** (2026-09-18, single-pass retrieval over the frozen
pipeline, after the ground-truth page audit and chart-filter re-index):

| Type | P@5 | R@10 | nDCG@10 | MRR |
|---|---|---|---|---|
| simple_factual (26) | 0.846 | 0.923 | 0.842 | 0.873 |
| multi_step (39) | 0.705 | 0.829 | 0.655 | 0.675 |
| edge_cases (15) | 0.767 | 0.767 | 0.608 | 0.611 |
| pooled (80) | 0.762 | 0.848 | 0.707 | 0.727 |

Runs are highly reproducible with the current rewriter prompt (two
consecutive runs differ by ≤0.02 at type level; 75/80 rewrites identical).
Five queries (Q008, Q012, Q014, Q070, Q076) score zero in the baseline run;
Q008, Q012, and Q076 are stable misses across rewriter variants — the target
set where System B's multi-pass retrieval is hypothesized to help.

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

## Usage

```bash
# Ingest the corpus (parse + chunk + embed + index; embedding needs a GPU
# session — see "GPU compute")
python scripts/ingest.py

# Rebuild the index from pre-computed vectors (no GPU)
python scripts/ingest.py --vectors-file data/vectors.json

# Run the retrieval benchmark over all 95 queries (~12 min)
python scripts/validate_retrieval.py
# → data/results/retrieval/validation_<timestamp>.json

# Tests
pytest
```

Utility scripts: `scripts/export_pages_txt.py` exports per-page text dumps for
ground-truth annotation/auditing; `scripts/inspect_table_pages.py` shows how
the parser and chunker render individual pages.

## Project structure

```
src/ingestion/   — PDF parsing, chunking, Qdrant indexing
src/retrieval/   — embedding, hybrid search, reranking, query rewriting
cluster/         — GPU server (FastAPI) and SLURM scripts for GWDG
scripts/         — ingest, validation, and utility entry points
tests/           — pytest suite
docs/            — design documents (chunking strategy, code standards)
config/          — settings.yaml
data/            — documents, queries, results (mostly gitignored)
```

## GPU compute (GWDG / KISSKI)

BGE-M3 and BGE-Reranker-v2-m3 run on GWDG's Grete GPU cluster (project
`kisski-bingen-ma`). Batch embedding jobs run on KISSKI (A100); interactive
model-server sessions run on Grete. See `cluster/README.md` for the full
workflow (submit → status → tunnel → run → teardown), `scripts/check_gpu.sh`
for job status, and `scripts/tunnel.sh` for the SSH tunnel helper.

The `gpu.remote_url` setting in `config/settings.yaml` (or the
`GPU_REMOTE_URL` env var) switches between local and remote GPU mode.

## Running on another machine (Docker)

Works on Linux/macOS/Windows. The image contains **only the orchestration
code** — no torch, no model weights: embedding/reranking always runs on a GPU
server (see `cluster/`), reached via `GPU_REMOTE_URL`.

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
