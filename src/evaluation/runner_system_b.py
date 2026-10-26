import logging

import anyio
from pydantic import BaseModel

from src.evaluation.ragas_scorer import RagasScores
from src.evaluation.runner_base import (
    EvaluationRunnerBase,
    Query,
    RetrievedChunkRecord,
    SummaryMetrics,
    _LATENCY_PRECISION,
    _utc_now,
)
from src.saia_rate_limiter import SaiaRateLimiter
from src.settings import Settings
from src.systems.system_b.pipeline import SystemBPipeline
from src.systems.system_b.types import RetrievalCallRecord, SystemBResult

logger = logging.getLogger(__name__)

_SYSTEM = "B"


class SystemBAnswerRecord(BaseModel):
    query_id: str
    query_type: str
    system: str
    run: int
    question: str
    source_document: str
    answer: str
    final_chunks: list[RetrievedChunkRecord]
    retrieval_calls: list[RetrievalCallRecord]
    retrieval_call_count: int
    llm_call_count: int
    warnings: list[str]
    latency_ms: int
    timestamp: str


class SystemBPerQueryResult(BaseModel):
    query_id: str
    query_type: str
    question: str
    source_document: str
    answer: str
    final_chunks: list[RetrievedChunkRecord]
    retrieval_queries: list[str]
    retrieval_call_count: int
    llm_call_count: int
    warnings: list[str]
    latency_ms_runs: list[int]
    latency_p50_ms: float
    ragas: RagasScores | None = None


class SystemBEvaluationFinal(BaseModel):
    system: str
    run_timestamp: str
    git_commit: str
    saia_model: str
    summary: dict[str, SummaryMetrics | None]
    per_query: list[SystemBPerQueryResult]


class SystemBEvaluationRunner(EvaluationRunnerBase):
    system = _SYSTEM
    _answer_record_model = SystemBAnswerRecord

    def __init__(
        self,
        cfg: Settings,
        rate_limiter: SaiaRateLimiter | None = None,
    ) -> None:
        system_b = cfg.system_b
        super().__init__(
            cfg,
            system_b.results_dir,
            system_b.answers_filename,
            system_b.ragas_scores_filename,
            system_b.evaluation_final_filename,
            system_b.runs_per_query,
            rate_limiter,
        )
        self._pipeline: SystemBPipeline | None = None

    def generate(self) -> None:
        anyio.run(self._generate_async)

    async def _generate_async(self) -> None:
        queries = self._load_queries()
        self._results_dir.mkdir(parents=True, exist_ok=True)
        completed = self._load_completed_runs()
        runs = range(1, self._runs_per_query + 1)
        pipeline: SystemBPipeline | None = None

        try:
            for query in queries:
                missing_runs = [
                    run for run in runs
                    if (query.query_id, run) not in completed
                ]
                if not missing_runs:
                    logger.info("Skipping %s: all runs already present", query.query_id)
                    continue

                if pipeline is None:
                    pipeline = self._get_pipeline()
                    await pipeline.start()

                for run in missing_runs:
                    result = await pipeline.run(query.question)
                    record = self._build_record(query, run, result)
                    self._append_answer(record)
                    logger.info(
                        "Generated %s run %d/%d in %d ms",
                        query.query_id, run, self._runs_per_query,
                        result.latency_ms,
                    )

                await anyio.sleep(self._cfg.system_b.generation_pace_seconds)
        finally:
            if pipeline is not None:
                await pipeline.close()

    def _get_pipeline(self) -> SystemBPipeline:
        if self._pipeline is None:
            self._pipeline = SystemBPipeline(self._cfg, self._rate_limiter)
        return self._pipeline

    def _build_record(
        self, query: Query, run: int, result: SystemBResult
    ) -> SystemBAnswerRecord:
        return SystemBAnswerRecord(
            query_id=query.query_id,
            query_type=query.query_type,
            system=_SYSTEM,
            run=run,
            question=query.question,
            source_document=self._source_document(query),
            answer=result.answer,
            final_chunks=[self._chunk_record(chunk) for chunk in result.final_chunks],
            retrieval_calls=result.retrieval_calls,
            retrieval_call_count=result.retrieval_call_count,
            llm_call_count=result.llm_call_count,
            warnings=result.warnings,
            latency_ms=result.latency_ms,
            timestamp=_utc_now(),
        )

    def _record_contexts(self, record: SystemBAnswerRecord) -> list[str]:
        return [chunk.text for chunk in record.final_chunks]

    def _build_per_query(
        self,
        query: Query,
        query_runs: dict[int, BaseModel],
        ragas_payload: RagasScores | None,
        latency_runs: list[int],
        latency_p50: float,
    ) -> SystemBPerQueryResult:
        first: SystemBAnswerRecord = query_runs[1]
        return SystemBPerQueryResult(
            query_id=query.query_id,
            query_type=query.query_type,
            question=first.question,
            source_document=first.source_document,
            answer=first.answer,
            final_chunks=first.final_chunks,
            retrieval_queries=[call.query for call in first.retrieval_calls],
            retrieval_call_count=first.retrieval_call_count,
            llm_call_count=first.llm_call_count,
            warnings=first.warnings,
            latency_ms_runs=latency_runs,
            latency_p50_ms=round(latency_p50, _LATENCY_PRECISION),
            ragas=ragas_payload,
        )

    def _build_final(
        self,
        run_timestamp: str,
        git_commit: str,
        saia_model: str,
        summary: dict[str, SummaryMetrics | None],
        per_query: list[BaseModel],
    ) -> SystemBEvaluationFinal:
        return SystemBEvaluationFinal(
            system=_SYSTEM,
            run_timestamp=run_timestamp,
            git_commit=git_commit,
            saia_model=saia_model,
            summary=summary,
            per_query=per_query,
        )
