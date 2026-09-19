import logging
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
from src.types import RetrievedChunk

logger = logging.getLogger(__name__)

GENERATOR_SYSTEM_PROMPT = (
    "You are a financial analyst assistant.\n"
    "Answer the question using only the numbered passages provided.\n"
    "Cite passage numbers inline, e.g. [1] or [2][3].\n"
    "If the passages do not contain the answer, say so plainly.\n"
    "Never fabricate figures, names, or dates."
)

_RETRYABLE_ERRORS = (
    InternalServerError,
    RateLimitError,
    APIConnectionError,
    APITimeoutError,
)


class SystemAGenerator:
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

    def generate(self, question: str, passages: list[RetrievedChunk]) -> str:
        """Generate a cited answer from retrieved passages."""
        user_prompt = self._build_user_prompt(question, passages)
        return self._call_llm_with_retry(user_prompt)

    def _build_user_prompt(
        self, question: str, passages: list[RetrievedChunk]
    ) -> str:
        if not passages:
            passages_block = "No passages were retrieved."
        else:
            passages_block = "\n\n".join(
                f"[{idx}] ({chunk.doc_name}, p.{chunk.page_number}, "
                f'section: "{chunk.section_title}") {chunk.text}'
                for idx, chunk in enumerate(passages, start=1)
            )
        return f"Question: {question}\n\nPassages:\n{passages_block}"

    def _call_llm_with_retry(self, user_prompt: str) -> str:
        attempts = self._cfg.evaluation.generator_retry_attempts
        last_exc: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                self._rate_limiter.acquire()
                response = self._client.chat.completions.create(
                    model=self._cfg.saia.model,
                    temperature=self._cfg.saia.temperature,
                    max_tokens=self._cfg.saia.max_tokens,
                    messages=[
                        {"role": "system", "content": GENERATOR_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    # qwen3.8 emits reasoning tokens before the answer; leaving
                    # thinking on would silently consume the generation budget
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                choice = response.choices[0]
                if choice.finish_reason == "length":
                    raise ValueError("Generated answer reached max_tokens and was truncated")
                content = choice.message.content
                self._validate_output(content)
                return content.strip()
            except _RETRYABLE_ERRORS as exc:
                last_exc = exc
                if attempt < attempts:
                    logger.warning(
                        "SAIA generation error on attempt %d/%d, retrying in %ds",
                        attempt, attempts, int(self._cfg.evaluation.generator_retry_wait_seconds),
                    )
                    time.sleep(self._cfg.evaluation.generator_retry_wait_seconds)
        assert last_exc is not None
        raise last_exc

    def _validate_output(self, content: str | None) -> None:
        if content is None:
            raise ValueError("SAIA generation returned empty content")
        stripped = content.strip()
        if not stripped:
            raise ValueError("SAIA generation returned whitespace-only content")
        if len(stripped) < self._cfg.evaluation.generator_min_answer_chars:
            raise ValueError(
                f"SAIA generation returned an answer under "
                f"{self._cfg.evaluation.generator_min_answer_chars} characters"
            )
