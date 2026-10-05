from unittest.mock import MagicMock, patch

import pytest
from openai import InternalServerError

from src.saia_rate_limiter import SaiaRateLimiter
from src.settings import Settings
from src.systems.system_a.generator import (
    GENERATOR_SYSTEM_PROMPT,
    SystemAGenerator,
)
from src.systems.system_a.pipeline import SystemAPipeline
from src.types import RetrievedChunk


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
        text="A retrieved passage about financial stability.",
        score=0.5,
        rerank_score=0.9,
    )


def _response(content: str, finish_reason: str = "stop") -> MagicMock:
    return MagicMock(
        choices=[
            MagicMock(
                finish_reason=finish_reason,
                message=MagicMock(content=content),
            )
        ]
    )


def _generator(limiter: SaiaRateLimiter | None = None) -> SystemAGenerator:
    return SystemAGenerator(Settings(), limiter or SaiaRateLimiter(28, 180, 900))


def test_build_user_prompt_includes_numbered_passages():
    generator = _generator()
    prompt = generator._build_user_prompt("Question?", [_chunk()])
    assert "Question: Question?" in prompt
    assert "[1] (doc.pdf, p.7, section: \"Section\")" in prompt
    assert "A retrieved passage about financial stability." in prompt


def test_build_user_prompt_handles_empty_passages():
    generator = _generator()
    prompt = generator._build_user_prompt("Question?", [])
    assert "No passages were retrieved." in prompt


def test_generate_returns_stripped_answer_and_acquires_limit():
    limiter = MagicMock(spec=SaiaRateLimiter)
    with patch("src.systems.system_a.generator.OpenAI") as mock_openai:
        mock_client = mock_openai.return_value
        mock_client.chat.completions.create.return_value = _response(
            "  This is a valid answer.  "
        )
        generator = _generator(limiter)
        answer = generator.generate("Question?", [_chunk()])

    assert answer == "This is a valid answer."
    limiter.acquire.assert_called_once()
    call = mock_client.chat.completions.create.call_args
    assert call.kwargs["model"] == "qwen3.8-27b"
    assert call.kwargs["temperature"] == 0
    assert call.kwargs["max_tokens"] == 2048
    assert call.kwargs["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }
    system_message = call.kwargs["messages"][0]
    assert system_message == {"role": "system", "content": GENERATOR_SYSTEM_PROMPT}


def test_generate_retries_transient_error():
    limiter = MagicMock(spec=SaiaRateLimiter)
    error = InternalServerError(
        "boom", response=MagicMock(status_code=500), body=None
    )
    with patch("src.systems.system_a.generator.OpenAI") as mock_openai, patch(
        "src.systems.system_a.generator.time.sleep"
    ) as mock_sleep:
        mock_client = mock_openai.return_value
        mock_client.chat.completions.create.side_effect = [
            error,
            _response("This is a valid answer."),
        ]
        generator = _generator(limiter)
        answer = generator.generate("Question?", [_chunk()])

    assert answer == "This is a valid answer."
    assert mock_client.chat.completions.create.call_count == 2
    assert limiter.acquire.call_count == 2
    mock_sleep.assert_called_once_with(30.0)


def test_generate_raises_after_retry_budget():
    limiter = MagicMock(spec=SaiaRateLimiter)
    error = InternalServerError(
        "boom", response=MagicMock(status_code=500), body=None
    )
    with patch("src.systems.system_a.generator.OpenAI") as mock_openai, patch(
        "src.systems.system_a.generator.time.sleep"
    ):
        mock_client = mock_openai.return_value
        mock_client.chat.completions.create.side_effect = error
        generator = _generator(limiter)
        with pytest.raises(InternalServerError):
            generator.generate("Question?", [_chunk()])

    assert mock_client.chat.completions.create.call_count == 5
    assert limiter.acquire.call_count == 5


def test_generate_rejects_empty_content():
    with patch("src.systems.system_a.generator.OpenAI") as mock_openai:
        mock_client = mock_openai.return_value
        mock_client.chat.completions.create.return_value = _response(None)
        generator = _generator(MagicMock(spec=SaiaRateLimiter))
        with pytest.raises(ValueError, match="empty content"):
            generator.generate("Question?", [_chunk()])


def test_generate_rejects_whitespace_only_content():
    with patch("src.systems.system_a.generator.OpenAI") as mock_openai:
        mock_client = mock_openai.return_value
        mock_client.chat.completions.create.return_value = _response("   \n")
        generator = _generator(MagicMock(spec=SaiaRateLimiter))
        with pytest.raises(ValueError, match="whitespace-only"):
            generator.generate("Question?", [_chunk()])


def test_generate_rejects_answer_below_minimum_length():
    with patch("src.systems.system_a.generator.OpenAI") as mock_openai:
        mock_client = mock_openai.return_value
        mock_client.chat.completions.create.return_value = _response("Too short")
        generator = _generator(MagicMock(spec=SaiaRateLimiter))
        with pytest.raises(ValueError, match="under 15 characters"):
            generator.generate("Question?", [_chunk()])


def test_generate_rejects_truncated_answer():
    with patch("src.systems.system_a.generator.OpenAI") as mock_openai:
        mock_client = mock_openai.return_value
        mock_client.chat.completions.create.return_value = _response(
            "A sufficiently long answer.", finish_reason="length"
        )
        generator = _generator(MagicMock(spec=SaiaRateLimiter))
        with pytest.raises(ValueError, match="truncated"):
            generator.generate("Question?", [_chunk()])


def test_pipeline_returns_rewritten_query_and_chunks():
    cfg = Settings()
    limiter = MagicMock(spec=SaiaRateLimiter)
    chunk = _chunk()
    with patch("src.systems.system_a.pipeline.Retriever") as mock_retriever_cls, patch(
        "src.systems.system_a.pipeline.SystemAGenerator"
    ) as mock_generator_cls:
        mock_retriever = mock_retriever_cls.return_value
        mock_retriever._rewriter.rewrite.return_value = "rewritten query"
        mock_retriever.retrieve.return_value = [chunk]
        mock_generator = mock_generator_cls.return_value
        mock_generator.generate.return_value = "Answer"

        pipeline = SystemAPipeline(cfg, limiter)
        result = pipeline.run("original question")

    assert result.answer == "Answer"
    assert result.retrieved_chunks == [chunk]
    assert result.rewritten_query == "rewritten query"
    assert result.latency_ms >= 0
    mock_retriever_cls.assert_called_once_with(cfg)
    mock_generator_cls.assert_called_once_with(cfg, limiter)
    limiter.acquire.assert_called_once_with()


def test_pipeline_only_reserves_rewrite_per_unique_question():
    cfg = Settings()
    limiter = MagicMock(spec=SaiaRateLimiter)
    with patch("src.systems.system_a.pipeline.Retriever") as mock_retriever_cls, patch(
        "src.systems.system_a.pipeline.SystemAGenerator"
    ):
        mock_retriever = mock_retriever_cls.return_value
        mock_retriever._rewriter.rewrite.return_value = "rewritten query"
        mock_retriever.retrieve.return_value = [_chunk()]

        pipeline = SystemAPipeline(cfg, limiter)
        pipeline.run("question")
        pipeline.run("question")
        pipeline.run("other question")

    assert limiter.acquire.call_count == 2
