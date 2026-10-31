import asyncio
import time

from src.saia_rate_limiter import SaiaRateLimiter
from src.settings import Settings


def test_from_settings_uses_saia_safety_limits():
    cfg = Settings()
    limiter = SaiaRateLimiter.from_settings(cfg)
    assert limiter._per_minute == 28
    assert limiter._per_hour == 180
    assert limiter._per_day == 450


def test_acquire_records_calls():
    limiter = SaiaRateLimiter(10, 10, 10)
    limiter.acquire()
    assert len(limiter._minute_calls) == 1
    assert len(limiter._hour_calls) == 1
    assert len(limiter._day_calls) == 1


def test_wait_when_minute_limit_reached():
    limiter = SaiaRateLimiter(2, 100, 100)
    limiter.acquire()
    limiter.acquire()
    assert limiter._wait_seconds_locked() > 0


def test_no_wait_after_minute_window_expires():
    limiter = SaiaRateLimiter(2, 100, 100)
    now = time.monotonic()
    limiter._minute_calls.extend([now - 61.0, now - 61.0])
    assert limiter._wait_seconds_locked() == 0.0


def test_purge_removes_expired_calls():
    limiter = SaiaRateLimiter(2, 2, 2)
    now = time.monotonic()
    limiter._minute_calls.append(now - 61.0)
    limiter._hour_calls.append(now - 3601.0)
    limiter._day_calls.append(now - 86401.0)

    limiter._purge_locked(now)

    assert len(limiter._minute_calls) == 0
    assert len(limiter._hour_calls) == 0
    assert len(limiter._day_calls) == 0


def test_acquire_waits_until_minute_slot_frees(monkeypatch):
    limiter = SaiaRateLimiter(1, 100, 100)
    limiter.acquire()

    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", sleeps.append)
    monkeypatch.setattr(
        limiter,
        "_wait_seconds_locked",
        lambda: 1.0 if not sleeps else 0.0,
    )

    limiter.acquire()

    assert sleeps == [1.0]
    assert len(limiter._minute_calls) == 2


def test_acquire_async_uses_async_sleep(monkeypatch):
    limiter = SaiaRateLimiter(1, 100, 100)
    limiter.acquire()

    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(
        limiter,
        "_wait_seconds_locked",
        lambda: 1.0 if not sleeps else 0.0,
    )

    asyncio.run(limiter.acquire_async())

    assert sleeps == [1.0]
    assert len(limiter._minute_calls) == 2
