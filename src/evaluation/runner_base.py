import json
import logging
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from src.evaluation.ragas_scorer import RagasScorer, RagasScores, ScoringSample
from src.evaluation.spaced_rate_limiter import SpacedRateLimiter
from src.saia_rate_limiter import SaiaRateLimiter
from src.settings import Settings
from src.types import RetrievedChunk

logger = logging.getLogger(__name__)

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


class SummaryMetrics(RagasScores):
    latency_p50_ms: float


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


class EvaluationRunnerBase:
    """Shared resume-safe generate/score/merge machinery for both systems."""

    system: str
    _answer_record_model: type[BaseModel]

    def __init__(
        self,
        cfg: Settings,
        results_dir: str,
        answers_filename: str,
        ragas_scores_filename: str,
        evaluation_final_filename: str,
        runs_per_query: int,
        rate_limiter: SaiaRateLimiter | None = None,
    ) -> None:
        self._cfg = cfg
        self._results_dir = Path(results_dir)
        self._answers_path = self._results_dir / answers_filename
        self._ragas_scores_path = self._results_dir / ragas_scores_filename
        self._final_path = self._results_dir / evaluation_final_filename
        self._runs_per_query = runs_per_query
        self._rate_limiter = rate_limiter or SaiaRateLimiter.from_settings(cfg)

    def score(self) -> None:
        queries = self._load_queries()
        first_runs = self._first_run_records(self._load_answer_records())
        scores: dict[str, RagasScores | None] = self._load_ragas_scores()
        scorer: RagasScorer | None = None

        for query in queries:
            if self._budget_exhausted():
                logger.info(
                    "Session SAIA budget reached; stopping score cleanly — resume next session"
                )
                return
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
                # RAGAS fires its per-metric judge calls in a parallel burst;
                # use a min-interval limiter (separate from the generate
                # limiter) so the burst stays under SAIA's 2-per-10s cap.
                scorer = RagasScorer(
                    self._cfg,
                    SpacedRateLimiter.from_settings(self._cfg),
                )

            sample = ScoringSample(
                query_id=query.query_id,
                question=record.question,
                answer=record.answer,
                retrieved_contexts=self._record_contexts(record),
                reference=query.ground_truth,
            )
            result = scorer.score(sample)
            scores[query.query_id] = result
            self._save_ragas_scores(scores)
            logger.info("Scored %s: %s", query.query_id, result.model_dump())

    def _budget_exhausted(self) -> bool:
        """Whether a per-session SAIA budget (if configured) is exhausted."""
        return False

    def merge(self) -> None:
        queries = self._load_queries()
        records = self._load_answer_records()
        scores = self._load_ragas_scores()
        runs = range(1, self._runs_per_query + 1)

        grouped: dict[str, dict[int, BaseModel]] = {}
        for record in records:
            grouped.setdefault(record.query_id, {}).setdefault(record.run, record)

        buckets: dict[str, dict[str, list[float]]] = {
            qtype: {**{metric: [] for metric in _RAGAS_METRICS}, "latency_p50_ms": []}
            for qtype in _QUERY_TYPES
            if qtype != "no_retrieval"
        }
        per_query: list[BaseModel] = []
        problems: list[str] = []

        for query in queries:
            query_runs = grouped.get(query.query_id, {})
            if any(run not in query_runs for run in runs):
                problems.append(f"{query.query_id} missing runs")
                continue

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
                self._build_per_query(query, query_runs, ragas_payload, latency_runs, latency_p50)
            )

        if problems:
            raise RuntimeError(f"Cannot merge incomplete evaluation: {problems}")

        self._results_dir.mkdir(parents=True, exist_ok=True)
        output = self._build_final(
            run_timestamp=_utc_now(),
            git_commit=_git_commit(),
            saia_model=self._cfg.saia.model,
            summary=self._build_summary(buckets),
            per_query=per_query,
        )
        self._final_path.write_text(
            output.model_dump_json(indent=2),
            encoding="utf-8",
        )
        logger.info("Evaluation written to %s", self._final_path)

    def _record_contexts(self, record: BaseModel) -> list[str]:
        raise NotImplementedError

    def _build_per_query(
        self,
        query: Query,
        query_runs: dict[int, BaseModel],
        ragas_payload: RagasScores | None,
        latency_runs: list[int],
        latency_p50: float,
    ) -> BaseModel:
        raise NotImplementedError

    def _build_final(
        self,
        run_timestamp: str,
        git_commit: str,
        saia_model: str,
        summary: dict[str, SummaryMetrics | None],
        per_query: list[BaseModel],
    ) -> BaseModel:
        raise NotImplementedError

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

    def _load_answer_records(self) -> list[BaseModel]:
        if not self._answers_path.exists():
            return []
        records: list[BaseModel] = []
        with self._answers_path.open(encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(self._answer_record_model.model_validate_json(line))
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid answer record at {self._answers_path} line {line_no}"
                    ) from exc
        return records

    def _append_answer(self, record: BaseModel) -> None:
        with self._answers_path.open("a", encoding="utf-8") as f:
            f.write(record.model_dump_json())
            f.write("\n")
            f.flush()

    def _first_run_records(
        self, records: list[BaseModel]
    ) -> dict[str, BaseModel]:
        first_runs: dict[str, BaseModel] = {}
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
