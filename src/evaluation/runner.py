import json
import logging
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from src.evaluation.ragas_scorer import RagasScorer, RagasScores, ScoringSample
from src.saia_rate_limiter import SaiaRateLimiter
from src.settings import Settings
from src.systems.system_a.pipeline import SystemAPipeline
from src.types import RetrievedChunk

logger = logging.getLogger(__name__)

_SYSTEM = "A"
_QUERY_TYPES = ("simple_factual", "multi_step", "no_retrieval", "edge_cases")
_RAGAS_METRICS = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
)
_SCORE_PRECISION = 4
_LATENCY_PRECISION = 1


class Query(BaseModel):
    model_config = ConfigDict(extra="ignore")

    query_id: str
    query_type: str
    question: str
    ground_truth: str
    source_document: str | None = None
    source_documents: dict[str, str] | None = None


class RetrievedChunkRecord(BaseModel):
    chunk_id: str
    doc_name: str
    page_number: int
    section_title: str
    chunk_type: str
    score: float
    text: str


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


class SummaryMetrics(RagasScores):
    latency_p50_ms: float


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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError) as exc:
        logger.warning("Could not determine git commit: %s", exc)
        return "unknown"


class EvaluationRunner:
    def __init__(
        self,
        cfg: Settings,
        rate_limiter: SaiaRateLimiter | None = None,
    ) -> None:
        self._cfg = cfg
        self._results_dir = Path(cfg.evaluation.system_a_results_dir)
        self._answers_path = self._results_dir / cfg.evaluation.answers_filename
        self._ragas_scores_path = self._results_dir / cfg.evaluation.ragas_scores_filename
        self._final_path = self._results_dir / cfg.evaluation.evaluation_final_filename
        self._rate_limiter = rate_limiter or SaiaRateLimiter.from_settings(cfg)
        self._pipeline: SystemAPipeline | None = None

    def generate(self) -> None:
        queries = self._load_queries()
        self._results_dir.mkdir(parents=True, exist_ok=True)
        completed = self._load_completed_runs()
        pipeline: SystemAPipeline | None = None
        runs = range(1, self._cfg.evaluation.runs_per_query + 1)

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
                    query.query_id, run, self._cfg.evaluation.runs_per_query,
                    result.latency_ms,
                )

            time.sleep(self._cfg.evaluation.generation_pace_seconds)

    def score(self) -> None:
        queries = self._load_queries()
        first_runs = self._first_run_records(self._load_answer_records())
        scores: dict[str, RagasScores | None] = self._load_ragas_scores()
        scorer: RagasScorer | None = None

        for query in queries:
            if query.query_type == "no_retrieval":
                scores[query.query_id] = None
                self._save_ragas_scores(scores)
                continue

            if not self._scores_have_nulls(scores.get(query.query_id)):
                logger.info("Skipping %s: RAGAS scores already present", query.query_id)
                continue

            if query.query_id not in first_runs:
                raise RuntimeError(f"Missing run 1 answer for {query.query_id}")
            record = first_runs[query.query_id]

            if scorer is None:
                scorer = RagasScorer(self._cfg, self._rate_limiter)

            sample = ScoringSample(
                query_id=query.query_id,
                question=record.question,
                answer=record.answer,
                retrieved_contexts=[chunk.text for chunk in record.retrieved_chunks],
                reference=query.ground_truth,
            )
            result = scorer.score(sample)
            scores[query.query_id] = result
            self._save_ragas_scores(scores)
            logger.info("Scored %s: %s", query.query_id, result.model_dump())

    def merge(self) -> None:
        queries = self._load_queries()
        records = self._load_answer_records()
        scores = self._load_ragas_scores()
        runs = range(1, self._cfg.evaluation.runs_per_query + 1)

        grouped: dict[str, dict[int, AnswerRecord]] = {}
        for record in records:
            grouped.setdefault(record.query_id, {}).setdefault(record.run, record)

        buckets: dict[str, dict[str, list[float]]] = {
            qtype: {**{metric: [] for metric in _RAGAS_METRICS}, "latency_p50_ms": []}
            for qtype in _QUERY_TYPES
            if qtype != "no_retrieval"
        }
        per_query: list[PerQueryResult] = []
        problems: list[str] = []

        for query in queries:
            query_runs = grouped.get(query.query_id, {})
            if any(run not in query_runs for run in runs):
                problems.append(f"{query.query_id} missing runs")
                continue

            first = query_runs[1]
            latency_runs = [query_runs[run].latency_ms for run in runs]
            latency_p50 = float(statistics.median(latency_runs))

            ragas_payload: RagasScores | None = None
            if query.query_type != "no_retrieval":
                query_scores = scores.get(query.query_id)
                if self._scores_have_nulls(query_scores):
                    problems.append(f"{query.query_id} missing RAGAS scores")
                    continue
                assert query_scores is not None
                ragas_payload = query_scores
                for metric in _RAGAS_METRICS:
                    buckets[query.query_type][metric].append(
                        float(getattr(query_scores, metric))
                    )
                buckets[query.query_type]["latency_p50_ms"].append(latency_p50)

            per_query.append(
                PerQueryResult(
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
            )

        if problems:
            raise RuntimeError(f"Cannot merge incomplete evaluation: {problems}")

        output = EvaluationFinal(
            system=_SYSTEM,
            run_timestamp=_utc_now(),
            git_commit=_git_commit(),
            saia_model=self._cfg.saia.model,
            summary=self._build_summary(buckets),
            per_query=per_query,
        )

        self._results_dir.mkdir(parents=True, exist_ok=True)
        self._final_path.write_text(
            output.model_dump_json(indent=2),
            encoding="utf-8",
        )
        logger.info("Evaluation written to %s", self._final_path)

    def _get_pipeline(self) -> SystemAPipeline:
        if self._pipeline is None:
            self._pipeline = SystemAPipeline(self._cfg, self._rate_limiter)
        return self._pipeline

    def _load_queries(self) -> list[Query]:
        queries_path = Path(self._cfg.data.queries_path)
        if not queries_path.exists():
            raise FileNotFoundError(f"Queries file not found: {queries_path}")
        raw = json.loads(queries_path.read_text(encoding="utf-8"))
        return [Query.model_validate(query) for query in raw]

    def _load_completed_runs(self) -> set[tuple[str, int]]:
        completed: set[tuple[str, int]] = set()
        for record in self._load_answer_records():
            completed.add((record.query_id, record.run))
        return completed

    def _load_answer_records(self) -> list[AnswerRecord]:
        if not self._answers_path.exists():
            return []
        records: list[AnswerRecord] = []
        with self._answers_path.open(encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(AnswerRecord.model_validate_json(line))
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid answer record at {self._answers_path} line {line_no}"
                    ) from exc
        return records

    def _append_answer(self, record: AnswerRecord) -> None:
        with self._answers_path.open("a", encoding="utf-8") as f:
            f.write(record.model_dump_json())
            f.write("\n")
            f.flush()

    def _first_run_records(
        self, records: list[AnswerRecord]
    ) -> dict[str, AnswerRecord]:
        first_runs: dict[str, AnswerRecord] = {}
        for record in records:
            if record.run == 1:
                first_runs.setdefault(record.query_id, record)
        return first_runs

    def _load_ragas_scores(self) -> dict[str, RagasScores | None]:
        if not self._ragas_scores_path.exists():
            return {}
        raw = json.loads(self._ragas_scores_path.read_text(encoding="utf-8"))
        return {
            query_id: (
                RagasScores.model_validate(value) if value is not None else None
            )
            for query_id, value in raw.items()
        }

    def _save_ragas_scores(
        self, scores: dict[str, RagasScores | None]
    ) -> None:
        self._results_dir.mkdir(parents=True, exist_ok=True)
        self._ragas_scores_path.write_text(
            json.dumps(
                {
                    query_id: (score.model_dump() if score is not None else None)
                    for query_id, score in scores.items()
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    @staticmethod
    def _source_document(query: Query) -> str:
        if query.source_document:
            return query.source_document
        if query.source_documents:
            return next(iter(query.source_documents.values()))
        return ""

    @staticmethod
    def _chunk_record(chunk: RetrievedChunk) -> RetrievedChunkRecord:
        return RetrievedChunkRecord(
            chunk_id=chunk.chunk_id,
            doc_name=chunk.doc_name,
            page_number=chunk.page_number,
            section_title=chunk.section_title,
            chunk_type=chunk.chunk_type,
            score=chunk.rerank_score,
            text=chunk.text,
        )

    @staticmethod
    def _scores_have_nulls(scores: RagasScores | None) -> bool:
        return scores is None or scores.has_nulls()

    @staticmethod
    def _build_summary(
        buckets: dict[str, dict[str, list[float]]],
    ) -> dict[str, SummaryMetrics | None]:
        summary: dict[str, SummaryMetrics | None] = {}
        for qtype in _QUERY_TYPES:
            if qtype == "no_retrieval":
                summary[qtype] = None
                continue

            bucket = buckets.get(qtype)
            if not bucket or not bucket["latency_p50_ms"]:
                summary[qtype] = None
                continue

            metrics = {
                metric: round(
                    sum(bucket[metric]) / len(bucket[metric]), _SCORE_PRECISION
                )
                for metric in _RAGAS_METRICS
            }
            metrics["latency_p50_ms"] = round(
                statistics.median(bucket["latency_p50_ms"]), _LATENCY_PRECISION
            )
            summary[qtype] = SummaryMetrics(**metrics)
        return summary
