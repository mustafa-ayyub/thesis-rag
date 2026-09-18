import logging
import time

from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)

from src.settings import Settings

logger = logging.getLogger(__name__)

REWRITER_SYSTEM_PROMPT = (
    "You are a search query optimizer for financial stability report retrieval. "
    "Rewrite the user's question as a concise, keyword-rich search query. "
    "Keep the core entities, metrics, and time references. Preserve technical "
    "terms and abbreviations exactly as given (e.g. NPL, CET1, LCR), and keep "
    "quoted phrases verbatim. For multi-part questions, prioritize the terms "
    "that identify the answer passage — do not pad with every minor sub-topic. "
    "Do not add words that are not in the question. "
    "Output only the rewritten query. No explanation. No punctuation at the end."
)

# The SDK's own sub-second retries are too aggressive for a busy shared backend
_RETRY_ATTEMPTS = 5
_RETRY_WAIT_S = 30.0

_RETRYABLE_ERRORS = (
    InternalServerError,
    RateLimitError,
    APIConnectionError,
    APITimeoutError,
)


class QueryRewriter:
    def __init__(self, cfg: Settings) -> None:
        self._cfg = cfg
        self._client = OpenAI(
            api_key=cfg.saia.api_key,
            base_url=cfg.saia.base_url,
        )
        self._cache: dict[str, str] = {}

    def rewrite(self, query: str) -> str:
        """Rewrite query for retrieval. Caches results within a session."""
        if query in self._cache:
            logger.debug("Cache hit for query: %s", query)
            return self._cache[query]

        rewritten = self._call_llm_with_retry(query)
        logger.debug("Rewrite: %r → %r", query, rewritten)
        self._cache[query] = rewritten
        return rewritten

    def _call_llm_with_retry(self, query: str) -> str:
        last_exc: Exception | None = None
        for attempt in range(1, _RETRY_ATTEMPTS + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self._cfg.saia.model,
                    temperature=self._cfg.saia.temperature,
                    max_tokens=self._cfg.saia.rewriter_max_tokens,
                    messages=[
                        {"role": "system", "content": REWRITER_SYSTEM_PROMPT},
                        {"role": "user", "content": query},
                    ],
                    # qwen3.8 thinks before answering; a one-step rewrite gains
                    # nothing from it, and the reasoning tokens would eat the budget
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                # content is None on a filtered/empty response — fall back to the
                # original query so retrieval still runs
                content = response.choices[0].message.content or query
                return content.strip() or query
            except _RETRYABLE_ERRORS as exc:
                last_exc = exc
                if attempt < _RETRY_ATTEMPTS:
                    logger.warning(
                        "SAIA error on attempt %d/%d, retrying in %ds",
                        attempt, _RETRY_ATTEMPTS, int(_RETRY_WAIT_S),
                    )
                    time.sleep(_RETRY_WAIT_S)
        assert last_exc is not None
        raise last_exc
