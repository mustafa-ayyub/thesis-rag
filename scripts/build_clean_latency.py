"""Build clean_latency.json from the existing 3 runs per query (no re-run).

The generate-phase `latency_ms` is the full wall time of `pipeline.run()`,
which includes time the SAIA rate limiter spent sleeping to respect quotas.
Rate-limit sleep is strictly *additive* (it only ever adds time), so the
fastest of a query's 3 runs is the least-contaminated estimate of true
end-to-end latency.

We therefore report, per query:
  * min_latency_ms   -- the de-contaminated estimate (fastest of 3 runs)
  * median_latency_ms-- for reference (what naive merge would use)
  * runs_ms          -- all three raw runs, for full transparency
  * runs_over_60s    -- how many runs look rate-limited (> 60s)
  * contaminated     -- True if even the minimum run is > 60s, i.e. every run
                        was rate-limited and min-of-3 cannot de-contaminate it.

The min-of-3 number is a conservative UPPER BOUND on true latency, not the
true floor (the whole generation window was rate-limited, so even the fastest
run usually carried some wait). It is applied identically to both systems so
the A-vs-B comparison is apples-to-apples.

Writes data/results/<system>/clean_latency.json for each system. Reads only
answers.jsonl + queries.json; touches nothing else.
"""

import json
import logging
import statistics
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

QUERIES = Path("data/queries/queries.json")
SYSTEMS = {
    "system_a": Path("data/results/system_a"),
    "system_b": Path("data/results/system_b"),
}
CONTAMINATION_THRESHOLD_MS = 60_000
REMEASURED_METHOD = "remeasured-min"
SCORED_BUCKETS = ("simple_factual", "multi_step", "edge_cases")
METHOD_NOTE = (
    "min-of-3-runs; rate-limit wait is additive so the fastest run is the least "
    "contaminated. This is a conservative UPPER BOUND on true latency, applied "
    "identically to System A and System B. Queries where even the minimum run "
    "exceeds 60s are flagged 'contaminated' (every run was rate-limited; exclude "
    "from latency stats)."
)


def _load_runs(answers_path: Path) -> dict[str, dict[int, int]]:
    """Return per-query latency keyed by run from an answers.jsonl file."""
    runs: dict[str, dict[int, int]] = defaultdict(dict)
    for line in answers_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        runs[rec["query_id"]][rec["run"]] = rec["latency_ms"]
    return runs


def _entry_for_query(query_type: str, latency_runs: list[int]) -> dict:
    """Build a clean_latency entry from a query's raw latency runs."""
    vals = sorted(latency_runs)
    min_ms = vals[0]
    return {
        "query_type": query_type,
        "min_latency_ms": min_ms,
        "median_latency_ms": round(statistics.median(vals)),
        "runs_ms": vals,
        "n_runs": len(vals),
        "runs_over_60s": sum(1 for v in vals if v > CONTAMINATION_THRESHOLD_MS),
        "contaminated": min_ms > CONTAMINATION_THRESHOLD_MS,
    }


def _report_buckets(name: str, out_path: Path, entries: dict[str, dict], contaminated: list[str]) -> None:
    buckets: dict[str, list[int]] = defaultdict(list)
    for entry in entries.values():
        if entry["query_type"] == "no_retrieval" or entry["contaminated"]:
            continue
        buckets[entry["query_type"]].append(entry["min_latency_ms"])
    logger.info("%s: wrote %s (%d queries, %d contaminated)", name, out_path, len(entries), len(contaminated))
    if contaminated:
        logger.info("  contaminated (excluded): %s", contaminated)
    for bucket in SCORED_BUCKETS:
        if buckets[bucket]:
            logger.info(
                "  %-14s n=%2d  min-of-3 bucket median = %5.1fs",
                bucket,
                len(buckets[bucket]),
                statistics.median(buckets[bucket]) / 1000,
            )


def _load_remeasured_entries(path: Path) -> dict[str, dict]:
    """Return any live-re-measured query entries from an existing file.

    These were produced by a targeted live re-measurement (see
    scripts/remeasure_contaminated.py) and must survive a rebuild, so the
    builder preserves them instead of recomputing from the raw runs.
    """
    if not path.exists():
        return {}
    existing = json.loads(path.read_text(encoding="utf-8"))
    return {
        qid: entry
        for qid, entry in existing.get("queries", {}).items()
        if entry.get("method") == REMEASURED_METHOD
    }


def build_for_system(name: str, results_dir: Path, qtype: dict[str, str]) -> None:
    answers_path = results_dir / "answers.jsonl"
    if not answers_path.exists():
        logger.warning("%s: no answers.jsonl, skipping", name)
        return

    runs = _load_runs(answers_path)
    preserved = _load_remeasured_entries(results_dir / "clean_latency.json")
    entries: dict[str, dict] = {}
    for qid in sorted(runs):
        entries[qid] = preserved.get(qid) or _entry_for_query(
            qtype.get(qid, "unknown"), list(runs[qid].values())
        )
    contaminated = [qid for qid, entry in entries.items() if entry["contaminated"]]

    payload = {
        "_method": METHOD_NOTE,
        "_contamination_threshold_ms": CONTAMINATION_THRESHOLD_MS,
        "_contaminated_query_ids": contaminated,
        "queries": entries,
    }
    out_path = results_dir / "clean_latency.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _report_buckets(name, out_path, entries, contaminated)


def main() -> None:
    queries = json.loads(QUERIES.read_text(encoding="utf-8"))
    qtype = {q["query_id"]: q["query_type"] for q in queries}
    for name, results_dir in SYSTEMS.items():
        build_for_system(name, results_dir, qtype)


if __name__ == "__main__":
    main()
