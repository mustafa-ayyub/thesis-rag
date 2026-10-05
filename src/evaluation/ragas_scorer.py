import logging
import math
import time
from typing import Any

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.embeddings import Embeddings
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from langchain_openai import ChatOpenAI
from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)
from pydantic import BaseModel, PrivateAttr
from ragas import EvaluationDataset, evaluate
from ragas.dataset_schema import EvaluationResult
from ragas.exceptions import RagasException
from ragas.metrics import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)
from ragas.run_config import RunConfig

from src.retrieval.embedder import Embedder
from src.saia_rate_limiter import SaiaRateLimiter
from src.settings import Settings

logger = logging.getLogger(__name__)

_RETRYABLE_RAGAS_ERRORS = (
    RagasException,
    RateLimitError,
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    TimeoutError,
)


class ScoringSample(BaseModel):
    query_id: str
    question: str
    answer: str
    retrieved_contexts: list[str]
    reference: str

    def to_ragas_row(self) -> dict[str, object]:
        return {
            "user_input": self.question,
            "response": self.answer,
            "retrieved_contexts": self.retrieved_contexts,
            "reference": self.reference,
        }


class RagasScores(BaseModel):
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None

    def has_nulls(self) -> bool:
        return any(
            value is None
            for value in (
                self.faithfulness,
                self.answer_relevancy,
                self.context_precision,
                self.context_recall,
            )
        )


class _BgeM3Embeddings(Embeddings):
    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder

    def embed_query(self, text: str) -> list[float]:
        dense, _ = self._embedder.embed_query(text)
        return dense

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [dense for dense, _ in self._embedder.embed_batch(texts)]


class _RateLimitedChatOpenAI(ChatOpenAI):
    _rate_limiter: SaiaRateLimiter = PrivateAttr()

    def __init__(self, rate_limiter: SaiaRateLimiter, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._rate_limiter = rate_limiter

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self._rate_limiter.acquire()
        return super()._generate(
            messages, stop=stop, run_manager=run_manager, **kwargs
        )

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        await self._rate_limiter.acquire_async()
        return await super()._agenerate(
            messages, stop=stop, run_manager=run_manager, **kwargs
        )


class RagasScorer:
    def __init__(
        self,
        cfg: Settings,
        rate_limiter: SaiaRateLimiter | None = None,
    ) -> None:
        self._cfg = cfg
        self._rate_limiter = rate_limiter or SaiaRateLimiter.from_settings(cfg)
        try:
            embedder = Embedder(cfg)
        except ImportError as exc:
            raise RuntimeError(
                "RAGAS scoring reuses the project's BGE-M3 embedder. Set "
                "gpu.remote_url to a running GPU tunnel, or run in an "
                "environment with FlagEmbedding and a GPU available."
            ) from exc
        self._embeddings = _BgeM3Embeddings(embedder)
        self._llm = _RateLimitedChatOpenAI(
            self._rate_limiter,
            model=cfg.saia.model,
            api_key=cfg.saia.api_key,
            base_url=cfg.saia.base_url,
            temperature=cfg.saia.temperature,
            max_tokens=cfg.evaluation.ragas_max_tokens,
            max_retries=0,
            # qwen3.8 emits reasoning tokens before the judge answer; leaving
            # thinking on would consume the structured-output budget
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        self._metrics = [
            Faithfulness(),
            AnswerRelevancy(strictness=cfg.evaluation.ragas_strictness),
            ContextPrecision(),
            ContextRecall(),
        ]
        self._run_config = RunConfig(
            timeout=cfg.evaluation.ragas_timeout_seconds,
            max_retries=0,
        )

    def score(self, sample: ScoringSample) -> RagasScores:
        """Score one RAG sample with faithfulness, relevancy, precision, recall."""
        attempts = self._cfg.evaluation.ragas_retry_attempts
        for attempt in range(1, attempts + 1):
            try:
                result = evaluate(
                    dataset=EvaluationDataset.from_list([sample.to_ragas_row()]),
                    metrics=self._metrics,
                    llm=self._llm,
                    embeddings=self._embeddings,
                    run_config=self._run_config,
                    raise_exceptions=True,
                    show_progress=False,
                )
                scores = self._extract_scores(result)
                if not scores.has_nulls():
                    time.sleep(self._cfg.evaluation.ragas_pace_seconds)
                    return scores
                logger.warning(
                    "RAGAS returned null scores for %s on attempt %d/%d",
                    sample.query_id, attempt, attempts,
                )
            except _RETRYABLE_RAGAS_ERRORS as exc:
                logger.warning(
                    "RAGAS evaluation failed for %s on attempt %d/%d: %s",
                    sample.query_id, attempt, attempts, exc,
                )
            if attempt < attempts:
                time.sleep(self._cfg.evaluation.ragas_retry_base_seconds * attempt)

        logger.error(
            "RAGAS scoring failed for %s after %d attempts; recording nulls",
            sample.query_id, attempts,
        )
        return RagasScores()

    def _extract_scores(self, result: EvaluationResult) -> RagasScores:
        row = result.scores[0]
        return RagasScores(
            faithfulness=self._coerce_score(row.get("faithfulness")),
            answer_relevancy=self._coerce_score(row.get("answer_relevancy")),
            context_precision=self._coerce_score(row.get("context_precision")),
            context_recall=self._coerce_score(row.get("context_recall")),
        )

    def _coerce_score(self, value: object) -> float | None:
        if value is None:
            return None
        score = float(value)
        return None if math.isnan(score) else score
