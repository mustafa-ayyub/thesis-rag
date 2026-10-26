import json
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
from src.systems.system_b.evidence_pool import EvidencePool
from src.systems.system_b.types import AgentDecision

logger = logging.getLogger(__name__)

AGENT_SYSTEM_PROMPT = (
    "You are a retrieval agent for financial stability reports.\n"
    "Decide whether the current evidence pool is sufficient or whether one more targeted retrieval is needed.\n"
    "Return JSON only with fields: covered, missing, decision, next_query, selected_evidence_ids, reason.\n"
    "Use decision \"retrieve_more\" only when a different query can plausibly find missing information.\n"
    "Use decision \"stop\" when the evidence is sufficient or no useful query remains.\n"
    "When stopping, selected_evidence_ids contains the evidence IDs needed for the final answer, ordered by importance.\n"
    "Never repeat a previous retrieval query."
)

_RETRYABLE_ERRORS = (
    InternalServerError,
    RateLimitError,
    APIConnectionError,
    APITimeoutError,
)


def parse_decision(content: str) -> AgentDecision:
    if content is None:
        raise ValueError("Agent returned empty content")

    text = content.strip()
    if text.startswith("```"):
        text = text.removeprefix("```")
        if text.lower().startswith("json"):
            text = text[4:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("Agent response did not contain a JSON object") from None
        payload = json.loads(text[start : end + 1])

    return AgentDecision.model_validate(payload)


class SystemBAgent:
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
        )

    async def decide(
        self,
        question: str,
        pool: EvidencePool,
        retrieval_calls_used: int,
        feedback: str | None = None,
    ) -> tuple[AgentDecision, int]:
        user_prompt = self._build_prompt(question, pool, retrieval_calls_used, feedback)
        attempts = self._cfg.system_b.agent_retry_attempts
        last_exc: Exception | None = None

        for attempt in range(1, attempts + 1):
            try:
                await self._rate_limiter.acquire_async()
                response = await self._client.chat.completions.create(
                    model=self._cfg.saia.model,
                    temperature=self._cfg.saia.temperature,
                    max_tokens=self._cfg.system_b.agent_max_tokens,
                    messages=[
                        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                choice = response.choices[0]
                if choice.finish_reason == "length":
                    raise ValueError("Agent response reached max_tokens and was truncated")
                decision = parse_decision(choice.message.content)
                return decision, attempt

            except _RETRYABLE_ERRORS as exc:
                last_exc = exc
                logger.warning(
                    "System B agent error on attempt %d/%d: %s",
                    attempt,
                    attempts,
                    exc,
                )
            except ValueError as exc:
                last_exc = exc
                logger.warning(
                    "System B agent returned invalid output on attempt %d/%d: %s",
                    attempt,
                    attempts,
                    exc,
                )

            if attempt < attempts:
                await anyio.sleep(self._cfg.system_b.agent_retry_base_seconds * attempt)

        assert last_exc is not None
        raise last_exc

    def _build_prompt(
        self,
        question: str,
        pool: EvidencePool,
        retrieval_calls_used: int,
        feedback: str | None,
    ) -> str:
        previous_queries = "\n".join(f"- {query}" for query in pool.queries) or "- None"
        feedback_block = f"Feedback: {feedback}\n\n" if feedback else ""
        max_final_chunks = self._cfg.system_b.max_final_chunks

        return (
            f"Question: {question}\n\n"
            f"Retrieval calls used: {retrieval_calls_used} / {self._cfg.system_b.max_retrieval_calls}\n\n"
            f"Previous retrieval queries:\n{previous_queries}\n\n"
            f"Evidence pool:\n{pool.render()}\n\n"
            f"Maximum final chunks: {max_final_chunks}\n\n"
            f"{feedback_block}"
            "Return JSON only."
        )
