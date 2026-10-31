import logging

import anyio
from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    InternalServerError,
    RateLimitError,
)

from src.saia_rate_limiter import SaiaRateLimiter
from src.settings import Settings
from src.systems.system_a.generator import GENERATOR_SYSTEM_PROMPT
from src.systems.system_b.types import GenerationResult
from src.types import RetrievedChunk

logger = logging.getLogger(__name__)

_RETRYABLE_ERRORS = (
    InternalServerError,
    RateLimitError,
    APIConnectionError,
    APITimeoutError,
)


class SystemBGenerator:
    def __init__(
        self,
        cfg: Settings,
        rate_limiter: SaiaRateLimiter | None = None,
        client: AsyncOpenAI | None = None,
    ) -> None:
        self._cfg = cfg
        self._rate_limiter = rate_limiter or SaiaRateLimiter.from_settings(cfg)
        self._client = client or AsyncOpenAI(
            api_key=cfg.saia.api_key,
            base_url=cfg.saia.base_url,
            timeout=cfg.saia.timeout_seconds,
        )

    async def generate(
        self,
        question: str,
        passages: list[RetrievedChunk],
    ) -> GenerationResult:
        user_prompt = self._build_user_prompt(question, passages)
        attempts = self._cfg.system_b.generator_retry_attempts
        last_exc: Exception | None = None

        for attempt in range(1, attempts + 1):
            try:
                await self._rate_limiter.acquire_async()
                response = await self._client.chat.completions.create(
                    model=self._cfg.saia.model,
                    temperature=self._cfg.saia.temperature,
                    max_tokens=self._cfg.saia.max_tokens,
                    messages=[
                        {"role": "system", "content": GENERATOR_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                choice = response.choices[0]
                if choice.finish_reason == "length":
                    raise ValueError("Generated answer reached max_tokens and was truncated")
                content = choice.message.content
                self._validate_output(content)
                return GenerationResult(answer=content.strip(), llm_call_count=attempt)

            except _RETRYABLE_ERRORS as exc:
                last_exc = exc
                logger.warning(
                    "System B generation error on attempt %d/%d: %s",
                    attempt,
                    attempts,
                    exc,
                )
            except ValueError as exc:
                last_exc = exc
                logger.warning(
                    "System B generation returned invalid output on attempt %d/%d: %s",
                    attempt,
                    attempts,
                    exc,
                )

            if attempt < attempts:
                await anyio.sleep(self._cfg.system_b.generator_retry_wait_seconds)

        assert last_exc is not None
        raise last_exc

    def _build_user_prompt(
        self, question: str, passages: list[RetrievedChunk]
    ) -> str:
        if not passages:
            passages_block = "No passages were selected."
        else:
            passages_block = "\n\n".join(
                f"[{idx}] ({chunk.doc_name}, p.{chunk.page_number}, "
                f'section: "{chunk.section_title}") {chunk.text}'
                for idx, chunk in enumerate(passages, start=1)
            )
        return f"Question: {question}\n\nPassages:\n{passages_block}"

    def _validate_output(self, content: str | None) -> None:
        if content is None:
            raise ValueError("SAIA generation returned empty content")
        stripped = content.strip()
        if not stripped:
            raise ValueError("SAIA generation returned whitespace-only content")
        if len(stripped) < self._cfg.system_b.generator_min_answer_chars:
            raise ValueError(
                f"SAIA generation returned an answer under "
                f"{self._cfg.system_b.generator_min_answer_chars} characters"
            )
