"""Build the System A vs System B comparison — the thesis empirical core.

Joins each system's v2 evaluation final (RAGAS quality, with the partial-credit
context-precision) with its clean_latency.json (the fair, apples-to-apples
latency estimate). Writes data/results/comparison.json plus a human-readable
Markdown table at data/results/comparison.md.

Quality metrics come from evaluation_final_v2.json (per bucket, mean of the
scored queries). Latency comes from clean_latency.json — NOT the raw
latency_p50_ms in the final, which for System B is inflated by SAIA rate-limit
waiting. Both systems use the same clean-latency method (min-of-3 upper bound;
System B's 2 tail queries were live-re-measured), so the latency comparison is
apples-to-apples.

Reads only the four result files; writes only comparison.{json,md}.
"""

import json
import logging
import statistics
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

RESULTS = Path("data/results")
SYSTEMS = {"system_a": "System A (standard RAG)", "system_b": "System B (MCP agent)"}
BUCKETS = ("simple_factual", "multi_step", "edge_cases")
RAGAS_METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


def _load_json(path: Path) -> dict:
    """Load and parse a result file, raising a clear error if it is unreadable."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"missing result file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"malformed JSON in {path}: {exc}") from exc


def _load(system: str) -> tuple[dict, dict]:
    final = _load_json(RESULTS / system / "evaluation_final_v2.json")
    latency = _load_json(RESULTS / system / "clean_latency.json")
    return final, latency


def _no_retrieval_summary(final: dict, latency: dict) -> dict:
    """Operational stats for no_retrieval queries (System B routing).

    RAGAS is intentionally N/A (there is no retrieved context to score). What is
    measurable, and what matters for the thesis, is whether the agent correctly
    skips the retrieval pipeline and answers directly, and how fast that is.
    """
    nr = [q for q in final["per_query"] if q["query_type"] == "no_retrieval"]
    if not nr or "retrieval_call_count" not in nr[0]:
        # System A has no agent routing (always retrieves); no such stats apply.
        return {}
    retrieval_calls = [q["retrieval_call_count"] for q in nr]
    llm_calls = [q["llm_call_count"] for q in nr]
    clean_ms = [
        rec["min_latency_ms"]
        for rec in latency["queries"].values()
        if rec["query_type"] == "no_retrieval"
    ]
    skipped = sum(1 for c in retrieval_calls if c == 0)
    s = sorted(clean_ms)
    idx = max(0, round(0.95 * (len(s) - 1)))
    return {
        "n": len(nr),
        "retrieval_skipped": skipped,
        "skip_rate": round(skipped / len(nr), 4),
        "retrieval_call_count_max": max(retrieval_calls),
        "llm_call_count_median": round(statistics.median(llm_calls)),
        "clean_latency": {
            "median_ms": round(statistics.median(s)),
            "p95_ms": s[idx],
            "min_ms": s[0],
            "max_ms": s[-1],
        },
        "ragas": "N/A (no retrieved context to score)",
    }


def _latency_by_bucket(latency: dict) -> dict[str, dict]:
    """Median and p95 of clean per-query latency, per bucket (no_retrieval excluded)."""
    buckets: dict[str, list[int]] = defaultdict(list)
    for rec in latency["queries"].values():
        t = rec["query_type"]
        if t == "no_retrieval" or rec.get("contaminated"):
            continue
        buckets[t].append(rec["min_latency_ms"])
    out: dict[str, dict] = {}
    for t, vals in buckets.items():
        s = sorted(vals)
        idx = max(0, round(0.95 * (len(s) - 1)))
        out[t] = {
            "n": len(s),
            "median_ms": round(statistics.median(s)),
            "p95_ms": s[idx],
            "min_ms": s[0],
            "max_ms": s[-1],
        }
    return out


def build() -> None:
    data: dict[str, dict] = {}
    for system in SYSTEMS:
        final, latency = _load(system)
        lat = _latency_by_bucket(latency)
        per_bucket: dict[str, dict] = {}
        for b in BUCKETS:
            quality = {m: round(final["summary"][b][m], 4) for m in RAGAS_METRICS}
            per_bucket[b] = {
                "quality": quality,
                "clean_latency": lat.get(b, {}),
                "raw_latency_p50_ms": final["summary"][b].get("latency_p50_ms"),
            }
        data[system] = {
            "label": SYSTEMS[system],
            "saia_model": final.get("saia_model"),
            "buckets": per_bucket,
            "no_retrieval": _no_retrieval_summary(final, latency),
        }

    payload = {
        "_about": (
            "System A vs System B comparison. Quality from evaluation_final_v2.json "
            "(partial-credit context precision). Latency from clean_latency.json "
            "(min-of-3 upper bound, identical method for both systems; System B's 2 "
            "tail queries Q004/Q050 were live-re-measured). raw_latency_p50_ms is the "
            "uncorrected merge value, kept only for transparency — do not compare it "
            "across systems (System B's is inflated by SAIA rate-limit waiting)."
        ),
        "systems": data,
    }
    (RESULTS / "comparison.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_markdown(data)
    logger.info("Wrote %s", RESULTS / "comparison.json")
    logger.info("Wrote %s", RESULTS / "comparison.md")


def _bucket_table(bucket: str, a: dict, b: dict) -> list[str]:
    lines = [f"## {bucket}", "", "| metric | System A | System B |", "|---|---|---|"]
    for metric in RAGAS_METRICS:
        va = a["buckets"][bucket]["quality"][metric]
        vb = b["buckets"][bucket]["quality"][metric]
        lines.append(f"| {metric} | {va:.3f} | {vb:.3f} |")
    la = a["buckets"][bucket]["clean_latency"]
    lb = b["buckets"][bucket]["clean_latency"]
    lines.append(f"| latency median (clean) | {la['median_ms']/1000:.1f}s | {lb['median_ms']/1000:.1f}s |")
    lines.append(f"| latency p95 (clean) | {la['p95_ms']/1000:.1f}s | {lb['p95_ms']/1000:.1f}s |")
    lines.append(f"| n (scored) | {la['n']} | {lb['n']} |")
    lines.append("")
    return lines


def _no_retrieval_table(nr: dict) -> list[str]:
    lat = nr["clean_latency"]
    return [
        "## no_retrieval (System B routing)",
        "",
        "General-knowledge queries. RAGAS is N/A (no retrieved context to score). "
        "System A has no agent routing and always retrieves, so this is a "
        "System B-only capability.",
        "",
        "| metric | System B |",
        "|---|---|",
        f"| retrieval skipped | {nr['retrieval_skipped']}/{nr['n']} ({nr['skip_rate']*100:.0f}%) |",
        f"| max retrieval calls | {nr['retrieval_call_count_max']} |",
        f"| llm calls (median) | {nr['llm_call_count_median']} |",
        f"| latency median (clean) | {lat['median_ms']/1000:.1f}s |",
        f"| latency p95 (clean) | {lat['p95_ms']/1000:.1f}s |",
        f"| ragas | {nr['ragas']} |",
        "",
    ]


def _write_markdown(data: dict) -> None:
    a, b = data["system_a"], data["system_b"]
    lines = [
        "# System A vs System B — Evaluation Comparison",
        "",
        f"- **System A:** {a['label']}",
        f"- **System B:** {b['label']}",
        "",
        "Quality: partial-credit RAGAS (v2 context precision). "
        "Latency: clean (min-of-3 upper bound, same method both systems).",
        "",
    ]
    for bucket in BUCKETS:
        lines.extend(_bucket_table(bucket, a, b))
    if b["no_retrieval"]:
        lines.extend(_no_retrieval_table(b["no_retrieval"]))
    (RESULTS / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    build()
