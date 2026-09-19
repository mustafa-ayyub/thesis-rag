import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.evaluation.ragas_scorer import RagasScores
from src.evaluation.runner import AnswerRecord, EvaluationRunner
from src.settings import Settings
from src.systems.system_a.pipeline import PipelineResult
from src.types import RetrievedChunk


def _cfg(tmp_path: Path) -> Settings:
    cfg = Settings()
    cfg.data.queries_path = str(tmp_path / "queries.json")
    cfg.evaluation.system_a_results_dir = str(tmp_path / "results")
    return cfg


def _chunk() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="abc123",
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


def _answer_record(
    query_id: str,
    query_type: str,
    run: int,
    source_document: str = "doc.pdf",
    latency_ms: int = 100,
) -> AnswerRecord:
    return AnswerRecord(
        query_id=query_id,
        query_type=query_type,
        system="A",
        run=run,
        question=f"Question {query_id}",
        source_document=source_document,
        rewritten_query=f"Rewritten {query_id}",
        answer=f"Answer {query_id} run {run}",
        retrieved_chunks=[],
        latency_ms=latency_ms,
        timestamp="2026-09-19T00:00:00Z",
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
    (tmp_path / "queries.json").write_text(
        json.dumps(queries), encoding="utf-8"
    )


def _write_answers(tmp_path: Path, records: list[AnswerRecord]) -> None:
    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    with (results_dir / "answers.jsonl").open("w", encoding="utf-8") as f:
        for record in records:
            f.write(record.model_dump_json())
            f.write("\n")


def test_generate_skips_completed_runs_and_appends_missing(tmp_path, monkeypatch):
    _write_queries(tmp_path)
    cfg = _cfg(tmp_path)
    _write_answers(
        tmp_path,
        [
            _answer_record("Q001", "simple_factual", 1),
            _answer_record("Q002", "no_retrieval", 2),
        ],
    )

    runner = EvaluationRunner(cfg)
    mock_pipeline = MagicMock()
    mock_pipeline.run.return_value = PipelineResult(
        answer="New answer",
        retrieved_chunks=[_chunk()],
        latency_ms=123,
        rewritten_query="New rewrite",
    )
    monkeypatch.setattr(runner, "_get_pipeline", lambda: mock_pipeline)
    monkeypatch.setattr("src.evaluation.runner.time.sleep", lambda _seconds: None)

    runner.generate()

    records = runner._load_answer_records()
    by_run = {(record.query_id, record.run): record for record in records}
    assert by_run[("Q001", 1)].answer == "Answer Q001 run 1"
    assert by_run[("Q001", 2)].answer == "New answer"
    assert by_run[("Q001", 3)].answer == "New answer"
    assert by_run[("Q002", 1)].answer == "New answer"
    assert by_run[("Q002", 2)].answer == "Answer Q002 run 2"
    assert by_run[("Q002", 3)].answer == "New answer"
    assert mock_pipeline.run.call_count == 4


def test_score_uses_first_run_and_records_no_retrieval_as_null(
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
    monkeypatch.setattr("src.evaluation.runner.time.sleep", lambda _seconds: None)

    with patch("src.evaluation.runner.RagasScorer") as mock_scorer_cls:
        mock_scorer = mock_scorer_cls.return_value
        mock_scorer.score.return_value = RagasScores(
            faithfulness=1.0,
            answer_relevancy=0.9,
            context_precision=0.8,
            context_recall=0.7,
        )
        runner = EvaluationRunner(cfg)
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

    runner = EvaluationRunner(cfg)
    runner.merge()

    final = json.loads(
        (tmp_path / "results" / "evaluation_final.json").read_text(encoding="utf-8")
    )
    assert final["system"] == "A"
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
    assert final["per_query"][0]["latency_p50_ms"] == 200.0
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

    runner = EvaluationRunner(cfg)
    with pytest.raises(RuntimeError, match="Q001 missing runs"):
        runner.merge()


def test_merge_fails_when_ragas_score_is_null(tmp_path):
    _write_queries(tmp_path)
    cfg = _cfg(tmp_path)
    _write_answers(
        tmp_path,
        [
            _answer_record("Q001", "simple_factual", 1),
            _answer_record("Q001", "simple_factual", 2),
            _answer_record("Q001", "simple_factual", 3),
            _answer_record("Q002", "no_retrieval", 1),
            _answer_record("Q002", "no_retrieval", 2),
            _answer_record("Q002", "no_retrieval", 3),
        ],
    )
    (tmp_path / "results" / "ragas_scores.json").write_text(
        json.dumps(
            {
                "Q001": {
                    "faithfulness": None,
                    "answer_relevancy": 1.0,
                    "context_precision": 1.0,
                    "context_recall": 1.0,
                },
                "Q002": None,
            }
        ),
        encoding="utf-8",
    )

    runner = EvaluationRunner(cfg)
    with pytest.raises(RuntimeError, match="Q001 missing RAGAS scores"):
        runner.merge()


def test_load_answer_records_reports_invalid_line(tmp_path):
    cfg = _cfg(tmp_path)
    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "answers.jsonl").write_text("not-json\n", encoding="utf-8")

    runner = EvaluationRunner(cfg)
    with pytest.raises(ValueError, match="line 1"):
        runner._load_answer_records()
