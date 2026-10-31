import asyncio

from src.settings import Settings
from src.systems.system_b.session_limiter import SessionLimiter


def test_counts_sync_acquires():
    limiter = SessionLimiter(100, 100, 100)
    assert limiter.calls_made == 0
    limiter.acquire()
    limiter.acquire()
    assert limiter.calls_made == 2
    assert len(limiter._minute_calls) == 2


def test_counts_async_acquires():
    limiter = SessionLimiter(100, 100, 100)
    asyncio.run(limiter.acquire_async())
    asyncio.run(limiter.acquire_async())
    assert limiter.calls_made == 2


def test_from_settings_returns_session_limiter():
    limiter = SessionLimiter.from_settings(Settings())
    assert isinstance(limiter, SessionLimiter)
    assert limiter._per_minute == 28
    assert limiter._per_hour == 180
    assert limiter._per_day == 450
