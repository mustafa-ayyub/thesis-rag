import time

from src.saia_rate_limiter import SaiaRateLimiter
from src.settings import Settings


class SpacedRateLimiter(SaiaRateLimiter):
    """SaiaRateLimiter plus a hard minimum interval between consecutive calls.

    SAIA enforces a short-window cap (2 requests per 10s) that a sliding
    minute/hour/day window cannot prevent: RAGAS fires its per-metric judge
    calls in a tight parallel burst, so the burst busts the short-window cap
    even when the per-minute rate is fine. Enforcing >= min_interval_s between
    calls (serialized by the parent lock) keeps every 10s window under that cap.

    min_interval_seconds > 10.0 guarantees at most one request per 10s window.
    """

    def __init__(
        self,
        per_minute: int,
        per_hour: int,
        per_day: int,
        min_interval_seconds: float,
    ) -> None:
        super().__init__(per_minute=per_minute, per_hour=per_hour, per_day=per_day)
        self._min_interval_s = float(min_interval_seconds)
        self._last_call: float = 0.0

    @classmethod
    def from_settings(cls, cfg: Settings) -> "SpacedRateLimiter":
        return cls(
            per_minute=cfg.saia.rate_limit_per_minute,
            per_hour=cfg.saia.rate_limit_per_hour,
            per_day=cfg.saia.rate_limit_per_day,
            min_interval_seconds=cfg.evaluation.ragas_min_interval_seconds,
        )

    def _wait_seconds_locked(self) -> float:
        base_wait = super()._wait_seconds_locked()
        spacing_wait = self._last_call + self._min_interval_s - time.monotonic()
        return max(base_wait, spacing_wait)

    def _record_call_locked(self) -> None:
        super()._record_call_locked()
        self._last_call = time.monotonic()
