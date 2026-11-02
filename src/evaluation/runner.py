import logging
import time

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
from src.systems.system_a.pipeline import SystemAPipeline
from src.types import RetrievedChunk

logger = logging.getLogger(__name__)

_SYSTEM = "A"


class AnswerRecord(BaseModel):
    query_id: str
    query_type: str
    system: str
    run: int
    question: str
    source_document: str
    rewritten_query: str
    answer: str
    retrieved_chunks: list[RetrievedChunkRecord]
    latency_ms: int
    timestamp: str


class PerQueryResult(BaseModel):
    query_id: str
    query_type: str
    question: str
    source_document: str
    rewritten_query: str
    answer: str
    retrieved_chunks: list[RetrievedChunkRecord]
    latency_ms_runs: list[int]
    latency_p50_ms: float
    ragas: RagasScores | None = None


class EvaluationFinal(BaseModel):
    system: str
    run_timestamp: str
    git_commit: str
    saia_model: str
    summary: dict[str, SummaryMetrics | None]
    per_query: list[PerQueryResult]


class EvaluationRunner(EvaluationRunnerBase):
    system = _SYSTEM
    _answer_record_model = AnswerRecord

    def __init__(
        self,
        cfg: Settings,
        rate_limiter: SaiaRateLimiter | None = None,
    ) -> None:
        evaluation = cfg.evaluation
        super().__init__(
            cfg,
            evaluation.system_a_results_dir,
            evaluation.answers_filename,
            evaluation.ragas_scores_filename,
            evaluation.evaluation_final_filename,
            evaluation.runs_per_query,
            rate_limiter,
        )
        self._pipeline: SystemAPipeline | None = None

    def generate(self) -> None:
        queries = self._load_queries()
        self._results_dir.mkdir(parents=True, exist_ok=True)
        completed = self._load_completed_runs()
        pipeline: SystemAPipeline | None = None
        runs = range(1, self._runs_per_query + 1)

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
            assert pipeline is not None

            for run in missing_runs:
                result = pipeline.run(query.question)
                record = AnswerRecord(
                    query_id=query.query_id,
                    query_type=query.query_type,
                    system=_SYSTEM,
                    run=run,
                    question=query.question,
                    source_document=self._source_document(query),
                    rewritten_query=result.rewritten_query,
                    answer=result.answer,
                    retrieved_chunks=[
                        self._chunk_record(chunk) for chunk in result.retrieved_chunks
                    ],
                    latency_ms=result.latency_ms,
                    timestamp=_utc_now(),
                )
                self._append_answer(record)
                logger.info(
                    "Generated %s run %d/%d in %d ms",
                    query.query_id, run, self._runs_per_query,
                    result.latency_ms,
                )

            time.sleep(self._cfg.evaluation.generation_pace_seconds)

    def _get_pipeline(self) -> SystemAPipeline:
        if self._pipeline is None:
            self._pipeline = SystemAPipeline(self._cfg, self._rate_limiter)
        return self._pipeline

    def _record_contexts(self, record: AnswerRecord) -> list[str]:
        return [chunk.text for chunk in record.retrieved_chunks]

    def _build_per_query(
        self,
        query: Query,
        query_runs: dict[int, BaseModel],
        ragas_payload: RagasScores | None,
        latency_runs: list[int],
        latency_p50: float,
    ) -> PerQueryResult:
        first: AnswerRecord = query_runs[1]
        return PerQueryResult(
            query_id=query.query_id,
            query_type=query.query_type,
            question=first.question,
            source_document=first.source_document,
            rewritten_query=first.rewritten_query,
            answer=first.answer,
            retrieved_chunks=first.retrieved_chunks,
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
    ) -> EvaluationFinal:
        return EvaluationFinal(
            system=_SYSTEM,
            run_timestamp=run_timestamp,
            git_commit=git_commit,
            saia_model=saia_model,
            summary=summary,
            per_query=per_query,
        )
