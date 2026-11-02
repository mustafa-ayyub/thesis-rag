"""Measure System B's clean end-to-end latency, free of rate-limit waits.

The generate-phase ``latency_ms`` is the full wall time of ``pipeline.run()``,
including the SAIA rate limiter's sleeps. Those sleeps are an artefact of the
free SAIA tier, not inherent latency, so we subtract them:

    clean_ms = latency_ms - rate_limit_wait_ms

The subtraction is only exact when no other hidden delay occurred. A flaky GPU
tunnel can make the MCP client retry (adding untracked wall time), so each
retrieval is checked and any run with a retry is discarded and re-measured.

Results go to data/results/system_b/clean_latency.json. Resume-safe; stops
before SAIA's daily cap. Reads/writes only clean_latency.json.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from src.evaluation.spaced_rate_limiter import SpacedRateLimiter
from src.settings import Settings
from src.systems.system_b.mcp_client import McpRetrievalClient, McpToolError
from src.systems.system_b.pipeline import SystemBPipeline
from src.systems.system_b.types import SystemBResult
from src.types import RetrievedChunk

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

OUT = Path("data/results/system_b/clean_latency.json")
SAFE_CALL_CAP = 380          # stop before SAIA's real 400/day hard cap
MAX_ATTEMPTS_PER_QUERY = 4   # re-measure a query up to this many times if blips occur
RETRY_SUSPECT_THRESHOLD_MS = 30_000  # retrieval slower than this (net of wait) implies a tunnel retry


@dataclass(frozen=True, slots=True)
class MeasuredRun:
    """A single pipeline run with its latency broken down by source."""

    result: SystemBResult
    wait_ms: float
    retry_events: int

    @property
    def clean_ms(self) -> float:
        """End-to-end latency with rate-limit sleep removed."""
        return self.result.latency_ms - self.wait_ms


class RetryCountingMcpClient:
    """Wraps the real MCP client and counts retrieve() retries per run.

    A single logical retrieve() call that the underlying client had to retry
    (tunnel blip -> ConnectError -> restart) carries untracked wall-clock
    delay. We expose `retry_events` so the measurement can discard and redo
    any contaminated run. This is a measurement harness only; production code
    is untouched.
    """

    def __init__(self, cfg: Settings, limiter: SpacedRateLimiter) -> None:
        self._inner = McpRetrievalClient(cfg.system_b, limiter)
        self.retry_events = 0

    async def start(self) -> None:
        await self._inner.start()

    async def close(self) -> None:
        await self._inner.close()

    async def retrieve(self, query: str) -> list[RetrievedChunk]:
        before = self._inner._rate_limiter.accumulated_wait_ms  # noqa: SLF001
        started = time.perf_counter()
        chunks = await self._inner.retrieve(query)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        rate_limit_wait_ms = self._inner._rate_limiter.accumulated_wait_ms - before  # noqa: SLF001
        # Productive retrieval (embed + query + rerank) is a few seconds. Once
        # the rate-limit permit wait is removed, anything still large means the
        # MCP client retried/restarted after a tunnel blip -> contaminated run.
        if (elapsed_ms - rate_limit_wait_ms) > RETRY_SUSPECT_THRESHOLD_MS:
            self.retry_events += 1
        return chunks


async def measure_once(
    cfg: Settings, limiter: SpacedRateLimiter, question: str
) -> MeasuredRun:
    client = RetryCountingMcpClient(cfg, limiter)
    pipeline = SystemBPipeline(cfg, rate_limiter=limiter, mcp_client=client)
    await pipeline.start()
    try:
        limiter.reset_wait()
        result = await pipeline.run(question)
        return MeasuredRun(
            result=result,
            wait_ms=limiter.accumulated_wait_ms,
            retry_events=client.retry_events,
        )
    finally:
        await pipeline.close()


async def _measure_clean_run(
    cfg: Settings, limiter: SpacedRateLimiter, query: dict
) -> MeasuredRun | None:
    """Measure a query until a blip-free run is obtained, else return None.

    A run is accepted only when it had no MCP retry and a non-negative clean
    latency, so ``clean_ms = latency - wait`` is exact.
    """
    qid = query["query_id"]
    for attempt in range(1, MAX_ATTEMPTS_PER_QUERY + 1):
        run = await measure_once(cfg, limiter, query["question"])
        if run.retry_events == 0 and run.clean_ms >= 0:
            return run
        logger.info(
            "%s attempt %d: DISCARD (tunnel retries=%d, clean=%dms) — re-measuring",
            qid,
            attempt,
            run.retry_events,
            round(run.clean_ms),
        )
    return None


def _run_record(query: dict, run: MeasuredRun) -> dict:
    return {
        "query_type": query["query_type"],
        "latency_ms": run.result.latency_ms,
        "wait_ms": round(run.wait_ms),
        "clean_ms": round(run.clean_ms),
        "llm_call_count": run.result.llm_call_count,
        "retrieval_call_count": run.result.retrieval_call_count,
    }


def _load_json(path: Path) -> dict | list:
    """Load and parse a JSON file, raising a clear error if it is unreadable."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"required file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"malformed JSON in {path}: {exc}") from exc


def _load_existing_results() -> dict:
    if not OUT.exists():
        return {}
    results = _load_json(OUT)
    logger.info("resuming: %d queries already measured", len(results))
    return results


def _persist(results: dict, qid: str, record: dict) -> None:
    results[qid] = record
    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(
        "%s [%-14s] latency=%7dms wait=%7dms clean=%6dms llm=%d retr=%d",
        qid,
        record["query_type"],
        record["latency_ms"],
        record["wait_ms"],
        record["clean_ms"],
        record["llm_call_count"],
        record["retrieval_call_count"],
    )


async def main() -> None:
    cfg = Settings()
    queries = _load_json(Path(cfg.data.queries_path))
    results = _load_existing_results()

    limiter = SpacedRateLimiter.from_settings(cfg)
    for query in queries:
        qid = query["query_id"]
        if qid in results:
            continue
        if len(limiter._day_calls) >= SAFE_CALL_CAP:  # noqa: SLF001
            logger.info("SAFE_CALL_CAP reached; stopping cleanly — resume next session")
            break

        try:
            run = await _measure_clean_run(cfg, limiter, query)
        except (McpToolError, ValueError) as exc:
            logger.warning("%s: measurement failed (%s); skipping", qid, exc)
            continue

        if run is None:
            logger.warning("%s: no clean run in %d tries; skipping", qid, MAX_ATTEMPTS_PER_QUERY)
            continue
        _persist(results, qid, _run_record(query, run))

    logger.info("Done. %d/%d queries measured -> %s", len(results), len(queries), OUT)


if __name__ == "__main__":
    asyncio.run(main())
