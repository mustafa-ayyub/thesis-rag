import time
from dataclasses import dataclass

from src.retrieval.retriever import Retriever
from src.saia_rate_limiter import SaiaRateLimiter
from src.settings import Settings
from src.systems.system_a.generator import SystemAGenerator
from src.types import RetrievedChunk


@dataclass
class PipelineResult:
    answer: str
    retrieved_chunks: list[RetrievedChunk]
    latency_ms: int
    rewritten_query: str


class SystemAPipeline:
    def __init__(
        self,
        cfg: Settings,
        rate_limiter: SaiaRateLimiter | None = None,
    ) -> None:
        self._cfg = cfg
        self._rate_limiter = rate_limiter or SaiaRateLimiter.from_settings(cfg)
        self._retriever = Retriever(cfg)
        self._generator = SystemAGenerator(cfg, self._rate_limiter)
        self._rewritten_questions: set[str] = set()

    @property
    def rate_limiter(self) -> SaiaRateLimiter:
        return self._rate_limiter

    def run(self, question: str) -> PipelineResult:
        """Run one retrieval pass and one generation call."""
        started = time.perf_counter()

        # The frozen Retriever makes its own rewriter call; reserve a SAIA
        # permit here so the shared limiter counts both System A calls.
        if question not in self._rewritten_questions:
            self._rate_limiter.acquire()
        rewritten_query = self._retriever._rewriter.rewrite(question)
        self._rewritten_questions.add(question)

        retrieved_chunks = self._retriever.retrieve(question)
        answer = self._generator.generate(question, retrieved_chunks)
        latency_ms = int((time.perf_counter() - started) * 1000)
        return PipelineResult(
            answer=answer,
            retrieved_chunks=retrieved_chunks,
            latency_ms=latency_ms,
            rewritten_query=rewritten_query,
        )
