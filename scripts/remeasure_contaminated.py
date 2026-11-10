"""Re-measure clean latency for specific contaminated queries.

For the given query IDs, run the real System B pipeline several times, keep
only runs with no MCP retry (so ``clean_ms = latency - wait`` is exact), and
take the minimum clean latency across those runs. The matching entries in
data/results/system_b/clean_latency.json are patched in place: contaminated is
cleared and the entry is tagged ``method="remeasured-min"``. Every other entry
is left untouched.

Usage:
    GPU_REMOTE_URL=http://localhost:8000 PYTHONUNBUFFERED=1 \
        .venv/bin/python scripts/remeasure_contaminated.py Q004 Q050
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

from src.evaluation.spaced_rate_limiter import SpacedRateLimiter
from src.settings import Settings
from scripts.measure_clean_latency import (
    McpToolError,
    MeasuredRun,
    _load_json,
    measure_once,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

OUT = Path("data/results/system_b/clean_latency.json")
RUNS_PER_QUERY = 3        # take the MIN clean across this many runs (SAIA latency is noisy)
MAX_ATTEMPTS_PER_RUN = 8  # retry a run if a tunnel blip (MCP retry) contaminates it
LIGHT_SPACING_SECONDS = 4.0  # only a handful of calls total, so light spacing is safe


def _build_limiter(cfg: Settings) -> SpacedRateLimiter:
    """Build a lightly-spaced limiter for a short targeted re-measurement.

    Only a few queries run, so light spacing stays under SAIA's short-window
    cap while keeping the run fast. The ``clean_ms = latency - wait``
    subtraction is exact regardless of the spacing value; spacing only affects
    wall-clock speed.
    """
    return SpacedRateLimiter(
        per_minute=cfg.saia.rate_limit_per_minute,
        per_hour=cfg.saia.rate_limit_per_hour,
        per_day=cfg.saia.rate_limit_per_day,
        min_interval_seconds=LIGHT_SPACING_SECONDS,
    )


async def _measure_blip_free_run(
    cfg: Settings, limiter: SpacedRateLimiter, query: dict, run_index: int
) -> MeasuredRun | None:
    """Measure one query until a blip-free run is obtained, else return None."""
    qid = query["query_id"]
    for attempt in range(1, MAX_ATTEMPTS_PER_RUN + 1):
        run = await measure_once(cfg, limiter, query["question"])
        if run.retry_events == 0 and run.clean_ms >= 0:
            logger.info(
                "%s run %d: clean=%dms (latency=%dms wait=%dms llm=%d retr=%d)",
                qid,
                run_index,
                round(run.clean_ms),
                run.result.latency_ms,
                round(run.wait_ms),
                run.result.llm_call_count,
                run.result.retrieval_call_count,
            )
            return run
        logger.info("%s run %d attempt %d: DISCARD (retries=%d)", qid, run_index, attempt, run.retry_events)
    logger.warning("%s run %d: no blip-free run in %d tries; skipping run", qid, run_index, MAX_ATTEMPTS_PER_RUN)
    return None


async def _remeasure_query(
    cfg: Settings, limiter: SpacedRateLimiter, query: dict
) -> list[MeasuredRun]:
    """Collect the blip-free runs for a single query."""
    runs: list[MeasuredRun] = []
    for run_index in range(1, RUNS_PER_QUERY + 1):
        run = await _measure_blip_free_run(cfg, limiter, query, run_index)
        if run is not None:
            runs.append(run)
    return runs


def _patch_entry(entry: dict, query: dict, runs: list[MeasuredRun]) -> dict:
    """Return the clean_latency entry updated with the minimum clean run."""
    best = min(runs, key=lambda r: r.clean_ms)
    entry.update(
        {
            "query_type": query["query_type"],
            "min_latency_ms": round(best.clean_ms),
            "contaminated": False,
            "method": "remeasured-min",
            "remeasured_clean_runs_ms": sorted(round(r.clean_ms) for r in runs),
            "remeasured_latency_ms": best.result.latency_ms,
            "remeasured_wait_ms": round(best.wait_ms),
            "llm_call_count": best.result.llm_call_count,
            "retrieval_call_count": best.result.retrieval_call_count,
        }
    )
    return entry


def _refresh_contaminated(data: dict) -> None:
    data["_contaminated_query_ids"] = sorted(
        qid for qid, entry in data["queries"].items() if entry.get("contaminated")
    )


async def main() -> None:
    target_ids = sys.argv[1:]
    if not target_ids:
        logger.error("usage: remeasure_contaminated.py Q004 Q050 ...")
        return

    cfg = Settings()
    queries = {q["query_id"]: q for q in _load_json(Path(cfg.data.queries_path))}
    unknown = [qid for qid in target_ids if qid not in queries]
    if unknown:
        logger.error("unknown query ids: %s", unknown)
        return

    data = _load_json(OUT)
    limiter = _build_limiter(cfg)

    for qid in target_ids:
        try:
            runs = await _remeasure_query(cfg, limiter, queries[qid])
        except (McpToolError, ValueError) as exc:
            logger.warning("%s: re-measurement failed (%s); leaving as-is", qid, exc)
            continue
        if not runs:
            logger.warning("%s: NO clean runs obtained; leaving as-is", qid)
            continue
        entry = _patch_entry(data["queries"].setdefault(qid, {}), queries[qid], runs)
        _refresh_contaminated(data)
        OUT.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info(
            "%s [%s] FINAL clean (min of %d runs) = %dms",
            qid,
            queries[qid]["query_type"],
            len(runs),
            entry["min_latency_ms"],
        )

    logger.info("Remaining contaminated: %s", data["_contaminated_query_ids"])


if __name__ == "__main__":
    asyncio.run(main())
