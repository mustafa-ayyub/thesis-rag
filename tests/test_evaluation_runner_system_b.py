import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.evaluation.ragas_scorer import RagasScores
from src.evaluation.runner_system_b import (
    SystemBAnswerRecord,
    SystemBEvaluationRunner,
)
from src.settings import Settings, SystemBSettings
from src.systems.system_b.types import (
    RetrievalCallRecord,
    SystemBResult,
)
from src.types import RetrievedChunk


def _cfg(tmp_path: Path) -> Settings:
    cfg = Settings(
        system_b=SystemBSettings(
            results_dir=str(tmp_path / "results"),
            generation_pace_seconds=0,
        )
    )
    cfg.data.queries_path = str(tmp_path / "queries.json")
    return cfg


def _chunk(chunk_id: str = "abc123") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_name="doc.pdf",
        doc_title="Test document",
        page_number=7,
        chunk_type="text",
        section_title="Section",
        chapter="2 Test chapter",
        token_count=16,
        text="A retrieved passage.",
        score=0.5,
        rerank_score=0.9,
    )


def _system_b_result(question: str, latency_ms: int = 123) -> SystemBResult:
    return SystemBResult(
        question=question,
        answer=f"Answer for {question}",
        final_chunks=[_chunk()],
        all_chunks=[_chunk()],
        retrieval_calls=[
            RetrievalCallRecord(
                call_number=1,
                query="NPL ratio",
                new_evidence_ids=["E1"],
                duplicate_chunk_ids=[],
                latency_ms=50,
            )
        ],
        retrieval_call_count=1,
        llm_call_count=4,
        warnings=[],
        latency_ms=latency_ms,
    )


class FakePipeline:
    def __init__(self) -> None:
        self.started = False
        self.closed = False
        self.questions: list[str] = []
        self.fail_after: int | None = None

    async def start(self) -> None:
        self.started = True

    async def close(self) -> None:
        self.closed = True

    async def run(self, question: str) -> SystemBResult:
        self.questions.append(question)
        if self.fail_after is not None and len(self.questions) > self.fail_after:
            raise RuntimeError("pipeline exploded")
        return _system_b_result(question)


def _answer_record(
    query_id: str,
    query_type: str,
    run: int,
    source_document: str = "doc.pdf",
    latency_ms: int = 100,
) -> SystemBAnswerRecord:
    return SystemBAnswerRecord(
        query_id=query_id,
        query_type=query_type,
        system="B",
        run=run,
        question=f"Question {query_id}",
        source_document=source_document,
        answer=f"Answer {query_id} run {run}",
        final_chunks=[
            {
                "chunk_id": "abc123",
                "doc_name": "doc.pdf",
                "page_number": 7,
                "section_title": "Section",
                "chunk_type": "text",
                "score": 0.9,
                "text": "A retrieved passage.",
            }
        ],
        retrieval_calls=[
            {
                "call_number": 1,
                "query": "NPL ratio",
                "new_evidence_ids": ["E1"],
                "duplicate_chunk_ids": [],
                "latency_ms": 50,
            }
        ],
        retrieval_call_count=1,
        llm_call_count=4,
        warnings=[],
        latency_ms=latency_ms,
        timestamp="2026-10-17T00:00:00Z",
    )


def _write_queries(tmp_path: Path) -> None:
    queries = [
        {
            "query_id": "Q001",
            "query_type": "simple_factual",
            "question": "Question Q001",
            "ground_truth": "Ground truth Q001",
            "source_document": "doc.pdf",
        },
        {
            "query_id": "Q002",
            "query_type": "no_retrieval",
            "question": "Question Q002",
            "ground_truth": "Ground truth Q002",
            "source_document": None,
        },
    ]
    (tmp_path / "queries.json").write_text(json.dumps(queries), encoding="utf-8")


def _write_answers(tmp_path: Path, records: list[SystemBAnswerRecord]) -> None:
    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    with (results_dir / "answers.jsonl").open("w", encoding="utf-8") as f:
        for record in records:
            f.write(record.model_dump_json())
            f.write("\n")


def test_generate_skips_completed_runs_and_lifecycle(tmp_path, monkeypatch):
    _write_queries(tmp_path)
    cfg = _cfg(tmp_path)
    _write_answers(
        tmp_path,
        [
            _answer_record("Q001", "simple_factual", 1),
            _answer_record("Q002", "no_retrieval", 2),
        ],
    )

    runner = SystemBEvaluationRunner(cfg)
    fake_pipeline = FakePipeline()
    monkeypatch.setattr(runner, "_get_pipeline", lambda: fake_pipeline)

    runner.generate()

    records = runner._load_answer_records()
    by_run = {(record.query_id, record.run): record for record in records}
    assert by_run[("Q001", 1)].answer == "Answer Q001 run 1"
    assert by_run[("Q001", 2)].answer == "Answer for Question Q001"
    assert by_run[("Q001", 3)].answer == "Answer for Question Q001"
    assert by_run[("Q002", 1)].answer == "Answer for Question Q002"
    assert by_run[("Q002", 2)].answer == "Answer Q002 run 2"
    assert by_run[("Q002", 3)].answer == "Answer for Question Q002"
    assert fake_pipeline.questions == [
        "Question Q001", "Question Q001", "Question Q002", "Question Q002",
    ]
    assert fake_pipeline.started
    assert fake_pipeline.closed
    record = by_run[("Q001", 2)]
    assert record.system == "B"
    assert record.llm_call_count == 4
    assert record.retrieval_call_count == 1
    assert record.final_chunks[0].chunk_id == "abc123"


def test_generate_closes_pipeline_on_error(tmp_path, monkeypatch):
    _write_queries(tmp_path)
    cfg = _cfg(tmp_path)

    runner = SystemBEvaluationRunner(cfg)
    fake_pipeline = FakePipeline()
    fake_pipeline.fail_after = 1
    monkeypatch.setattr(runner, "_get_pipeline", lambda: fake_pipeline)

    with pytest.raises(RuntimeError, match="pipeline exploded"):
        runner.generate()

    assert fake_pipeline.started
    assert fake_pipeline.closed


def test_generate_no_pipeline_when_all_runs_present(tmp_path, monkeypatch):
    _write_queries(tmp_path)
    cfg = _cfg(tmp_path)
    _write_answers(
        tmp_path,
        [
            _answer_record("Q001", "simple_factual", run) for run in (1, 2, 3)
        ]
        + [
            _answer_record("Q002", "no_retrieval", run) for run in (1, 2, 3)
        ],
    )

    runner = SystemBEvaluationRunner(cfg)
    fake_pipeline = FakePipeline()
    monkeypatch.setattr(runner, "_get_pipeline", lambda: fake_pipeline)

    runner.generate()

    assert fake_pipeline.questions == []
    assert not fake_pipeline.started
    assert not fake_pipeline.closed


def test_score_uses_first_run_final_chunks_and_nulls_no_retrieval(
    tmp_path, monkeypatch
):
    _write_queries(tmp_path)
    cfg = _cfg(tmp_path)
    _write_answers(
        tmp_path,
        [
            _answer_record("Q001", "simple_factual", 1),
            _answer_record("Q002", "no_retrieval", 1),
        ],
    )

    with patch("src.evaluation.runner_base.RagasScorer") as mock_scorer_cls:
        mock_scorer = mock_scorer_cls.return_value
        mock_scorer.score.return_value = RagasScores(
            faithfulness=1.0,
            answer_relevancy=0.9,
            context_precision=0.8,
            context_recall=0.7,
        )
        runner = SystemBEvaluationRunner(cfg)
        runner.score()

    scores = runner._load_ragas_scores()
    assert scores["Q001"] == RagasScores(
        faithfulness=1.0,
        answer_relevancy=0.9,
        context_precision=0.8,
        context_recall=0.7,
    )
    assert scores["Q002"] is None
    assert mock_scorer.score.call_count == 1
    sample = mock_scorer.score.call_args.args[0]
    assert sample.query_id == "Q001"
    assert sample.reference == "Ground truth Q001"
    assert sample.retrieved_contexts == ["A retrieved passage."]


def test_merge_writes_final_evaluation(tmp_path):
    _write_queries(tmp_path)
    cfg = _cfg(tmp_path)
    _write_answers(
        tmp_path,
        [
            _answer_record("Q001", "simple_factual", 1, latency_ms=100),
            _answer_record("Q001", "simple_factual", 2, latency_ms=200),
            _answer_record("Q001", "simple_factual", 3, latency_ms=300),
            _answer_record("Q002", "no_retrieval", 1, latency_ms=100),
            _answer_record("Q002", "no_retrieval", 2, latency_ms=200),
            _answer_record("Q002", "no_retrieval", 3, latency_ms=300),
        ],
    )
    (tmp_path / "results" / "ragas_scores.json").write_text(
        json.dumps(
            {
                "Q001": {
                    "faithfulness": 1.0,
                    "answer_relevancy": 0.9,
                    "context_precision": 0.8,
                    "context_recall": 0.7,
                },
                "Q002": None,
            }
        ),
        encoding="utf-8",
    )

    runner = SystemBEvaluationRunner(cfg)
    runner.merge()

    final = json.loads(
        (tmp_path / "results" / "evaluation_final.json").read_text(encoding="utf-8")
    )
    assert final["system"] == "B"
    assert final["summary"]["simple_factual"] == {
        "faithfulness": 1.0,
        "answer_relevancy": 0.9,
        "context_precision": 0.8,
        "context_recall": 0.7,
        "latency_p50_ms": 200.0,
    }
    assert final["summary"]["no_retrieval"] is None
    assert [entry["query_id"] for entry in final["per_query"]] == ["Q001", "Q002"]
    assert final["per_query"][0]["latency_ms_runs"] == [100, 200, 300]
    assert final["per_query"][0]["retrieval_queries"] == ["NPL ratio"]
    assert final["per_query"][0]["retrieval_call_count"] == 1
    assert final["per_query"][0]["llm_call_count"] == 4
    assert final["per_query"][1]["ragas"] is None


def test_merge_fails_when_a_run_is_missing(tmp_path):
    _write_queries(tmp_path)
    cfg = _cfg(tmp_path)
    _write_answers(
        tmp_path,
        [
            _answer_record("Q001", "simple_factual", 1),
            _answer_record("Q001", "simple_factual", 2),
            _answer_record("Q002", "no_retrieval", 1),
            _answer_record("Q002", "no_retrieval", 2),
            _answer_record("Q002", "no_retrieval", 3),
        ],
    )
    (tmp_path / "results" / "ragas_scores.json").write_text(
        json.dumps(
            {
                "Q001": {
                    "faithfulness": 1.0,
                    "answer_relevancy": 1.0,
                    "context_precision": 1.0,
                    "context_recall": 1.0,
                },
                "Q002": None,
            }
        ),
        encoding="utf-8",
    )

    runner = SystemBEvaluationRunner(cfg)
    with pytest.raises(RuntimeError, match="Q001 missing runs"):
        runner.merge()


def test_load_answer_records_reports_invalid_line(tmp_path):
    cfg = _cfg(tmp_path)
    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "answers.jsonl").write_text("not-json\n", encoding="utf-8")

    runner = SystemBEvaluationRunner(cfg)
    with pytest.raises(ValueError, match="line 1"):
        runner._load_answer_records()
