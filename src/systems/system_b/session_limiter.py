import threading

from src.saia_rate_limiter import SaiaRateLimiter
from src.settings import Settings


class SessionLimiter(SaiaRateLimiter):
    """Shared SAIA limiter plus a per-process call counter for the daily batch.

    The counter bounds one evaluation session (one process). It resets on
    restart, which is exactly the scope the per-session SAIA budget targets.
    """

    def __init__(self, per_minute: int, per_hour: int, per_day: int) -> None:
        super().__init__(per_minute, per_hour, per_day)
        self._session_calls = 0
        self._session_lock = threading.Lock()

    @classmethod
    def from_settings(cls, cfg: Settings) -> "SessionLimiter":
        return cls(
            per_minute=cfg.saia.rate_limit_per_minute,
            per_hour=cfg.saia.rate_limit_per_hour,
            per_day=cfg.saia.rate_limit_per_day,
        )

    @property
    def calls_made(self) -> int:
        with self._session_lock:
            return self._session_calls

    def acquire(self) -> None:
        super().acquire()
        with self._session_lock:
            self._session_calls += 1

    async def acquire_async(self) -> None:
        await super().acquire_async()
        with self._session_lock:
            self._session_calls += 1
