import asyncio
import logging
import threading
import time
from collections import deque

from src.settings import Settings

logger = logging.getLogger(__name__)

_MINUTE_S = 60.0
_HOUR_S = 3600.0
_DAY_S = 86400.0


class SaiaRateLimiter:
    """Sliding-window limiter for SAIA's minute, hour, and daily quotas."""

    def __init__(
        self,
        per_minute: int,
        per_hour: int,
        per_day: int,
    ) -> None:
        self._per_minute = per_minute
        self._per_hour = per_hour
        self._per_day = per_day
        self._minute_calls: deque[float] = deque()
        self._hour_calls: deque[float] = deque()
        self._day_calls: deque[float] = deque()
        self._lock = threading.Lock()

    @classmethod
    def from_settings(cls, cfg: Settings) -> "SaiaRateLimiter":
        return cls(
            per_minute=cfg.saia.rate_limit_per_minute,
            per_hour=cfg.saia.rate_limit_per_hour,
            per_day=cfg.saia.rate_limit_per_day,
        )

    def acquire(self) -> None:
        while True:
            with self._lock:
                wait_s = self._wait_seconds_locked()
                if wait_s <= 0:
                    self._record_call_locked()
                    return
            logger.debug("SAIA rate limiter waiting %.1fs", wait_s)
            time.sleep(wait_s)

    async def acquire_async(self) -> None:
        while True:
            with self._lock:
                wait_s = self._wait_seconds_locked()
                if wait_s <= 0:
                    self._record_call_locked()
                    return
            logger.debug("SAIA rate limiter waiting %.1fs", wait_s)
            await asyncio.sleep(wait_s)

    def _wait_seconds_locked(self) -> float:
        now = time.monotonic()
        self._purge_locked(now)

        waits = [0.0]
        if len(self._minute_calls) >= self._per_minute:
            waits.append(self._minute_calls[0] + _MINUTE_S - now)
        if len(self._hour_calls) >= self._per_hour:
            waits.append(self._hour_calls[0] + _HOUR_S - now)
        if len(self._day_calls) >= self._per_day:
            waits.append(self._day_calls[0] + _DAY_S - now)
        return max(waits)

    def _record_call_locked(self) -> None:
        now = time.monotonic()
        self._minute_calls.append(now)
        self._hour_calls.append(now)
        self._day_calls.append(now)

    def _purge_locked(self, now: float) -> None:
        self._minute_calls = deque(
            ts for ts in self._minute_calls if ts > now - _MINUTE_S
        )
        self._hour_calls = deque(
            ts for ts in self._hour_calls if ts > now - _HOUR_S
        )
        self._day_calls = deque(
            ts for ts in self._day_calls if ts > now - _DAY_S
        )
