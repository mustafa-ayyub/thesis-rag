"""Resilient POST for the remote GPU model server.

The SSH tunnel to the cluster can drop transiently (network blips on the
login→compute path); a single 30 s read timeout would otherwise kill a
multi-hour validation run. Mirrors the SAIA retry decision: retry on
transport errors and 5xx, never on 4xx.
"""
import logging
import time

import httpx

logger = logging.getLogger(__name__)

_ATTEMPTS = 3
_WAIT_SECONDS = 5.0


def post_with_retry(url: str, payload: dict, timeout: float) -> httpx.Response:
    last: Exception | None = None
    for attempt in range(1, _ATTEMPTS + 1):
        try:
            resp = httpx.post(url, json=payload, timeout=timeout)
            if resp.status_code < 500:
                resp.raise_for_status()
                return resp
            last = httpx.HTTPStatusError(
                f"HTTP {resp.status_code}", request=resp.request, response=resp
            )
        except httpx.TransportError as exc:
            last = exc
        if attempt < _ATTEMPTS:
            logger.warning(
                "Remote call to %s failed (attempt %d/%d): %s",
                url, attempt, _ATTEMPTS, type(last).__name__,
            )
            time.sleep(_WAIT_SECONDS)
    assert last is not None
    raise last
