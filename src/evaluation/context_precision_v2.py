import json
import logging
import re
import time

from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)

from src.saia_rate_limiter import SaiaRateLimiter
from src.settings import Settings

logger = logging.getLogger(__name__)

_RETRYABLE_ERRORS = (
    InternalServerError,
    RateLimitError,
    APIConnectionError,
    APITimeoutError,
)

_ALLOWED_VERDICTS = (0.0, 0.5, 1.0)

SYSTEM_PROMPT = (
    "You are a retrieval evaluation judge for financial stability reports.\n"
    "Score each retrieved context for its usefulness to the reference answer.\n"
    "Use exactly one of these values for each context:\n"
    "- 1.0: the context directly contains the answer or fully supports the reference answer.\n"
    "- 0.5: the context is partially useful because it supports one part of a multi-part "
    "answer or gives important supporting evidence.\n"
    "- 0.0: the context is irrelevant or not useful.\n"
    "Evaluate each context independently.\n"
    "Return JSON only."
)


def weighted_average_precision(verdicts: list[float]) -> float:
    denominator = sum(verdicts)
    if denominator <= 0:
        return 0.0
    numerator = 0.0
    cumulative = 0.0
    for rank, verdict in enumerate(verdicts, start=1):
        cumulative += verdict
        numerator += (cumulative / rank) * verdict
    return numerator / denominator


def _snap_verdict(value: float) -> float:
    nearest = min(_ALLOWED_VERDICTS, key=lambda allowed: abs(allowed - value))
    if abs(nearest - value) > 0.26:
        raise ValueError(f"Verdict {value} is not close to 0, 0.5, or 1")
    return nearest


def _extract_verdicts(content: str, expected_count: int) -> list[float]:
    if content is None:
        raise ValueError("Empty judge response")

    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    values: object = None
    try:
        payload = json.loads(text)
        if isinstance(payload, dict) and "verdicts" in payload:
            values = payload["verdicts"]
        else:
            values = payload
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", text, flags=re.DOTALL)
        if not match:
            raise
        values = json.loads(match.group(0))

    if not isinstance(values, list) or len(values) != expected_count:
        raise ValueError(f"Expected {expected_count} verdicts")

    return [_snap_verdict(float(value)) for value in values]


class PartialCreditContextPrecision:
    def __init__(
        self,
        cfg: Settings,
        rate_limiter: SaiaRateLimiter | None = None,
    ) -> None:
        self._cfg = cfg
        self._rate_limiter = rate_limiter or SaiaRateLimiter.from_settings(cfg)
        self._client = OpenAI(
            api_key=cfg.saia.api_key,
            base_url=cfg.saia.base_url,
        )

    def score(
        self,
        question: str,
        reference: str,
        contexts: list[tuple[str, str]],
    ) -> tuple[float, list[float]]:
        context_block = "\n\n".join(
            f"Context {index}: ({source})\n{text}"
            for index, (source, text) in enumerate(contexts, start=1)
        )
        user_prompt = (
            f"Question: {question}\n\n"
            f"Reference answer: {reference}\n\n"
            f"Retrieved contexts:\n{context_block}\n\n"
            'Return exactly: {{"verdicts": [ ... ]}} with '
            f"{len(contexts)} numbers in the same order."
        )

        attempts = self._cfg.evaluation.ragas_retry_attempts
        last_exc: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                self._rate_limiter.acquire()
                response = self._client.chat.completions.create(
                    model=self._cfg.saia.model,
                    temperature=self._cfg.saia.temperature,
                    max_tokens=200,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                content = response.choices[0].message.content
                verdicts = _extract_verdicts(content, len(contexts))
                return weighted_average_precision(verdicts), verdicts
            except _RETRYABLE_ERRORS as exc:
                last_exc = exc
                logger.warning(
                    "Partial-credit context precision failed on attempt %d/%d: %s",
                    attempt,
                    attempts,
                    exc,
                )
            except ValueError as exc:
                last_exc = exc
                logger.warning(
                    "Partial-credit context precision got invalid output on attempt %d/%d: %s",
                    attempt,
                    attempts,
                    exc,
                )
            if attempt < attempts:
                time.sleep(self._cfg.evaluation.ragas_retry_base_seconds * attempt)

        assert last_exc is not None
        raise last_exc
