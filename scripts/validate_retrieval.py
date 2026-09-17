#!/usr/bin/env python3
"""Run the retriever against annotated queries and write validation metrics.

Queries file format (data/queries/queries.json):
    [
      {
        "query_id": "Q001",
        "query_type": "simple_factual",
        "question": "What are the main systemic risks in the euro area?",
        "source_document": "ECB-Financial-Stability-Reviews-2026.pdf",
        "ground_truth_pages": [32, 33]
      }
    ]

Single-document queries carry "source_document". A handful of by-design
cross-edition queries instead carry "source_documents" — a map of
ground-truth page -> document, e.g. {"76": "...-2023.pdf", "87": "...-2025.pdf"}.

Query types: simple_factual, multi_step, no_retrieval, edge_cases.
no_retrieval queries are skipped for metrics (summary entry is null).
Metrics match on (source_document, page_number) — the five corpus PDFs
share page numbers, so page alone is ambiguous.

P@5 is normalized by the number of relevant (doc, page) pairs (at most 5):
a query whose single answer page is found in the top 5 scores 1.0. Plain
hits/5 would cap a perfect single-page query at 0.2.

Usage:
    python scripts/validate_retrieval.py
    python scripts/validate_retrieval.py --config config/settings.yaml
    python scripts/validate_retrieval.py --queries data/queries/queries.json
"""
import argparse
import json
import logging
import math
import re
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

_QUERY_TYPES = ("simple_factual", "multi_step", "no_retrieval", "edge_cases")
_VALIDATION_TOP_N = 10


def _parse_ground_truth_pages(q: dict) -> list[int]:
    if "ground_truth_pages" in q:
        return q["ground_truth_pages"]
    source = q.get("source_page", "")
    return [int(n) for n in re.findall(r"\d+", source)]


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError) as exc:
        logger.warning("Could not determine git commit: %s", exc)
        return "unknown"


def _precision_at_k(pages: list[tuple[str, int]], relevant: set[tuple[str, int]], k: int) -> float:
    # Normalized precision: fraction of the query's relevant (doc, page) pairs
    # (at most k) found in the top k. A full hit is 1.0 regardless of how many
    # pages the query is grounded on — plain hits/k is capped by |relevant|/k.
    if not relevant:
        return 0.0
    top = pages[:k]
    hits = sum(1 for p in top if p in relevant)
    return hits / min(len(relevant), k)


def _recall_at_k(pages: list[tuple[str, int]], relevant: set[tuple[str, int]], k: int) -> float:
    if not relevant:
        return 0.0
    top = pages[:k]
    return sum(1 for p in top if p in relevant) / len(relevant)


def _ndcg_at_k(pages: list[tuple[str, int]], relevant: set[tuple[str, int]], k: int) -> float:
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, p in enumerate(pages[:k], start=1)
        if p in relevant
    )
    n_rel = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, n_rel + 1))
    return dcg / idcg if idcg > 0 else 0.0


def _mrr(pages: list[tuple[str, int]], relevant: set[tuple[str, int]]) -> float:
    for rank, p in enumerate(pages, start=1):
        if p in relevant:
            return 1.0 / rank
    return 0.0


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate retrieval pipeline.")
    parser.add_argument("--config", default="config/settings.yaml")
    parser.add_argument("--queries", default=None)
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.settings import load_settings
    from src.retrieval.retriever import Retriever

    cfg = load_settings(args.config)
    queries_path = Path(args.queries or cfg.data.queries_path)

    if not queries_path.exists():
        logger.error("Queries file not found: %s", queries_path)
        sys.exit(1)

    with queries_path.open() as f:
        queries: list[dict] = json.load(f)

    if not queries:
        logger.error("No queries in %s", queries_path)
        sys.exit(1)

    missing = [q.get("query_id", "?") for q in queries
               if q.get("query_type") != "no_retrieval"
               and not q.get("source_document") and not q.get("source_documents")]
    if missing:
        logger.error("Queries missing source_document/source_documents: %s", missing)
        sys.exit(1)

    logger.info("Loaded %d queries", len(queries))

    val_cfg = deepcopy(cfg)
    val_cfg.retrieval.rerank_top_n = _VALIDATION_TOP_N
    retriever = Retriever(val_cfg)

    per_query: list[dict] = []
    rewrite_latencies: list[float] = []
    by_type: dict[str, dict[str, list[float]]] = {
        t: {"precision_at_5": [], "recall_at_10": [], "ndcg_at_10": [], "mrr": []}
        for t in _QUERY_TYPES
        if t != "no_retrieval"
    }

    for q in queries:
        qid = q.get("query_id", "?")
        qtype = q.get("query_type", "")
        question = q.get("question", q.get("query", ""))
        ground_truth = _parse_ground_truth_pages(q)

        entry: dict = {
            "query_id": qid,
            "query_type": qtype,
            "question": question,
        }

        if qtype == "no_retrieval":
            per_query.append(entry)
            continue

        logger.info("Query %s (%s): %s", qid, qtype, question[:80])
        t0 = time.time()
        rewritten = retriever._rewriter.rewrite(question)
        rewrite_s = time.time() - t0
        rewrite_latencies.append(rewrite_s)
        chunks = retriever.retrieve(question)
        # all five PDFs share page numbers — a hit needs the right doc AND page;
        # dedupe so two chunks from the same page don't double-count
        retrieved = list(dict.fromkeys((c.doc_name, c.page_number) for c in chunks))
        doc_map = q.get("source_documents")
        if doc_map:
            relevant = {(doc_map[str(p)], p) for p in ground_truth}
        else:
            relevant = {(q["source_document"], p) for p in ground_truth}

        p5 = _precision_at_k(retrieved, relevant, 5)
        r10 = _recall_at_k(retrieved, relevant, 10)
        ndcg = _ndcg_at_k(retrieved, relevant, 10)
        mrr = _mrr(retrieved, relevant)

        if "source_documents" in q:
            entry["source_documents"] = q["source_documents"]
        else:
            entry["source_document"] = q["source_document"]
        entry.update({
            "rewritten_query": rewritten,
            "rewrite_seconds": round(rewrite_s, 2),
            "retrieved_docs": [d for d, _ in retrieved],
            "retrieved_pages": [p for _, p in retrieved],
            "ground_truth_pages": ground_truth,
            "precision_at_5": round(p5, 4),
            "recall_at_10": round(r10, 4),
            "ndcg_at_10": round(ndcg, 4),
            "mrr": round(mrr, 4),
        })
        per_query.append(entry)

        if qtype in by_type:
            by_type[qtype]["precision_at_5"].append(p5)
            by_type[qtype]["recall_at_10"].append(r10)
            by_type[qtype]["ndcg_at_10"].append(ndcg)
            by_type[qtype]["mrr"].append(mrr)

        logger.info(
            "  p@5=%.3f r@10=%.3f ndcg@10=%.3f mrr=%.3f",
            p5, r10, ndcg, mrr,
        )
        time.sleep(val_cfg.saia.request_interval_seconds)

    summary: dict = {}
    for qtype in _QUERY_TYPES:
        if qtype == "no_retrieval":
            summary[qtype] = None
            continue
        buckets = by_type[qtype]
        if not buckets["mrr"]:
            summary[qtype] = None
        else:
            summary[qtype] = {
                "precision_at_5": round(_avg(buckets["precision_at_5"]), 4),
                "recall_at_10": round(_avg(buckets["recall_at_10"]), 4),
                "ndcg_at_10": round(_avg(buckets["ndcg_at_10"]), 4),
                "mrr": round(_avg(buckets["mrr"]), 4),
            }

    output = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "summary": summary,
        "per_query": per_query,
    }

    results_dir = Path(cfg.data.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = results_dir / f"validation_{ts}.json"
    with out_path.open("w") as f:
        json.dump(output, f, indent=2)

    logger.info("Results written to %s", out_path)
    if rewrite_latencies:
        lat = sorted(rewrite_latencies)
        logger.info(
            "Rewriter latency: n=%d min=%.1fs median=%.1fs p90=%.1fs max=%.1fs (>10s: %d)",
            len(lat), lat[0], lat[len(lat) // 2], lat[int(len(lat) * 0.9)], lat[-1],
            sum(1 for x in lat if x > 10),
        )
    for qtype, metrics in summary.items():
        if metrics:
            logger.info(
                "  %-16s p@5=%.3f r@10=%.3f ndcg@10=%.3f mrr=%.3f",
                qtype, metrics["precision_at_5"], metrics["recall_at_10"],
                metrics["ndcg_at_10"], metrics["mrr"],
            )


if __name__ == "__main__":
    main()
