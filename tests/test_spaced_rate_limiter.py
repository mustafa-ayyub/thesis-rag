import asyncio
import time

from src.evaluation.spaced_rate_limiter import SpacedRateLimiter
from src.settings import Settings


def test_spaced_from_settings_uses_saia_and_interval():
    limiter = SpacedRateLimiter.from_settings(Settings())

    assert limiter._per_minute == 28
    assert limiter._per_hour == 180
    assert limiter._per_day == 450
    assert limiter._min_interval_s == 11.0


def test_spaced_wait_includes_spacing_after_call():
    limiter = SpacedRateLimiter(100, 100, 100, min_interval_seconds=10.0)
    limiter.acquire()

    # Right after a call, the spacing wait must be ~min_interval even though
    # the sliding minute/hour/day windows are far from full.
    assert limiter._wait_seconds_locked() > 5.0


def test_spaced_no_spacing_wait_after_interval_elapses():
    limiter = SpacedRateLimiter(100, 100, 100, min_interval_seconds=10.0)
    limiter.acquire()
    limiter._last_call = time.monotonic() - 11.0

    assert limiter._wait_seconds_locked() == 0.0


def test_spaced_serializes_sync_calls():
    limiter = SpacedRateLimiter(100, 100, 100, min_interval_seconds=0.3)
    start = time.monotonic()
    for _ in range(3):
        limiter.acquire()
    elapsed = time.monotonic() - start

    # 3 calls spaced 0.3s apart -> ~0.6s of enforced spacing.
    assert elapsed >= 0.5
    assert len(limiter._minute_calls) == 3


def test_spaced_serializes_concurrent_async_calls():
    # RAGAS fires its per-metric judge calls in parallel; the shared lock +
    # min-interval must serialize them so no 10s window exceeds the SAIA cap.
    limiter = SpacedRateLimiter(100, 100, 100, min_interval_seconds=0.3)

    async def run() -> None:
        await asyncio.gather(limiter.acquire_async(), limiter.acquire_async(), limiter.acquire_async())

    start = time.monotonic()
    asyncio.run(run())
    elapsed = time.monotonic() - start

    assert elapsed >= 0.5
    assert len(limiter._minute_calls) == 3
