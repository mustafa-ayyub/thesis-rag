import logging
import time
from typing import Any, Protocol

from src.saia_rate_limiter import SaiaRateLimiter
from src.settings import Settings
from src.systems.system_b.agent import SystemBAgent
from src.systems.system_b.evidence_pool import EvidencePool, RepeatedQueryError
from src.systems.system_b.generator import SystemBGenerator
from src.systems.system_b.mcp_client import McpRetrievalClient
from src.systems.system_b.types import (
    AgentDecision,
    GenerationResult,
    RetrievalCallRecord,
    SystemBResult,
)
from src.types import RetrievedChunk

logger = logging.getLogger(__name__)


class RetrievalTool(Protocol):
    async def retrieve(self, query: str) -> list[RetrievedChunk]: ...


class AgentProtocol(Protocol):
    async def decide(
        self,
        question: str,
        pool: EvidencePool,
        retrieval_calls_used: int,
        feedback: str | None,
    ) -> tuple[AgentDecision, int]: ...


class GeneratorProtocol(Protocol):
    async def generate(
        self,
        question: str,
        passages: list[RetrievedChunk],
    ) -> GenerationResult: ...


class SystemBPipeline:
    def __init__(
        self,
        cfg: Settings,
        rate_limiter: SaiaRateLimiter | None = None,
        mcp_client: RetrievalTool | None = None,
        agent: AgentProtocol | None = None,
        generator: GeneratorProtocol | None = None,
    ) -> None:
        self._cfg = cfg
        self._rate_limiter = rate_limiter or SaiaRateLimiter.from_settings(cfg)
        self._mcp_client: Any = mcp_client or McpRetrievalClient(cfg.system_b, self._rate_limiter)
        self._agent = agent or SystemBAgent(cfg, self._rate_limiter)
        self._generator = generator or SystemBGenerator(cfg, self._rate_limiter)

    async def start(self) -> None:
        start = getattr(self._mcp_client, "start", None)
        if callable(start):
            await start()

    async def close(self) -> None:
        close = getattr(self._mcp_client, "close", None)
        if callable(close):
            await close()

    async def run(self, question: str) -> SystemBResult:
        started = time.perf_counter()
        pool = EvidencePool(self._cfg.system_b.max_final_chunks)
        retrieval_calls: list[RetrievalCallRecord] = []
        decisions: list[AgentDecision] = []
        warnings: list[str] = []
        llm_call_count = 0
        feedback: str | None = None
        unusable_query_strikes = 0
        selected_chunks: list[RetrievedChunk] | None = None

        while selected_chunks is None:
            decision, decision_llm_calls = await self._agent.decide(
                question,
                pool,
                len(retrieval_calls),
                feedback,
            )
            decisions.append(decision)
            llm_call_count += decision_llm_calls

            if decision.decision == "stop":
                selected_chunks = self._select_with_fallback(
                    pool,
                    decision.selected_evidence_ids,
                    warnings,
                )
                break

            if len(retrieval_calls) >= self._cfg.system_b.max_retrieval_calls:
                warnings.append(
                    "Agent requested retrieval after the maximum number of calls; "
                    "finalizing with current evidence"
                )
                selected_chunks = self._select_with_fallback(
                    pool, [], warnings, force_fallback=True
                )
                break

            query = decision.next_query.strip()
            try:
                pool.mark_query(query)
            except RepeatedQueryError:
                feedback = (
                    "Your proposed query was already used. "
                    "Propose a different query or stop."
                )
                warnings.append(f"Agent proposed repeated query: {query}")
            except ValueError:
                feedback = (
                    "Your proposed query has no searchable words. "
                    "Propose a different query or stop."
                )
                warnings.append(f"Agent proposed unsearchable query: {query}")

            if feedback is not None:
                unusable_query_strikes += 1
                if unusable_query_strikes >= self._cfg.system_b.agent_retry_attempts:
                    warnings.append(
                        "Unusable query proposals persisted; "
                        "finalizing with current evidence"
                    )
                    selected_chunks = self._select_with_fallback(
                        pool, [], warnings, force_fallback=True
                    )
                    break
                continue

            unusable_query_strikes = 0
            feedback = None
            retrieval_started = time.perf_counter()
            chunks = await self._mcp_client.retrieve(query)
            retrieval_latency_ms = int((time.perf_counter() - retrieval_started) * 1000)
            new_evidence_ids, duplicate_chunk_ids = pool.add_chunks(
                len(retrieval_calls) + 1,
                chunks,
            )
            retrieval_calls.append(
                RetrievalCallRecord(
                    call_number=len(retrieval_calls) + 1,
                    query=query,
                    new_evidence_ids=new_evidence_ids,
                    duplicate_chunk_ids=duplicate_chunk_ids,
                    latency_ms=retrieval_latency_ms,
                )
            )

        generation = await self._generator.generate(question, selected_chunks)
        llm_call_count += generation.llm_call_count
        latency_ms = int((time.perf_counter() - started) * 1000)

        return SystemBResult(
            question=question,
            answer=generation.answer,
            final_chunks=selected_chunks,
            all_chunks=pool.all_chunks,
            retrieval_calls=retrieval_calls,
            agent_decisions=decisions,
            latency_ms=latency_ms,
            retrieval_call_count=len(retrieval_calls),
            llm_call_count=llm_call_count,
            warnings=warnings,
        )

    def _select_with_fallback(
        self,
        pool: EvidencePool,
        evidence_ids: list[str],
        warnings: list[str],
        force_fallback: bool = False,
    ) -> list[RetrievedChunk]:
        if not force_fallback and evidence_ids:
            try:
                return pool.select_chunks(evidence_ids)
            except ValueError as exc:
                warnings.append(f"Invalid agent selection: {exc}")
        elif not force_fallback and pool.evidence_ids:
            warnings.append(
                "Agent stopped without selecting evidence; "
                "using the first retrieved items instead"
            )

        fallback_ids = pool.evidence_ids[: self._cfg.system_b.max_final_chunks]
        if fallback_ids:
            warnings.append(
                f"Falling back to the first {len(fallback_ids)} retrieved evidence items"
            )
            return pool.select_chunks(fallback_ids)
        return []

    async def __aenter__(self) -> "SystemBPipeline":
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        await self.close()
