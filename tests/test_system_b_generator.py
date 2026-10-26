from unittest.mock import AsyncMock, MagicMock

import pytest
from openai import InternalServerError

from src.saia_rate_limiter import SaiaRateLimiter
from src.settings import Settings, SystemBSettings
from src.systems.system_a.generator import GENERATOR_SYSTEM_PROMPT
from src.systems.system_b.generator import SystemBGenerator
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
        choices=[MagicMock(finish_reason=finish_reason, message=MagicMock(content=content))]
    )


def _generator(
    client: MagicMock,
    limiter: MagicMock | None = None,
    cfg: Settings | None = None,
) -> SystemBGenerator:
    return SystemBGenerator(
        cfg or Settings(system_b=SystemBSettings(generator_retry_wait_seconds=0)),
        limiter or MagicMock(spec=SaiaRateLimiter),
        client,
    )


@pytest.mark.anyio
async def test_generator_returns_answer_and_call_count():
    limiter = MagicMock(spec=SaiaRateLimiter)
    limiter.acquire_async = AsyncMock()
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=_response("  This is a valid answer.  "))
    generator = _generator(client, limiter)

    result = await generator.generate("Question?", [_chunk()])

    assert result.answer == "This is a valid answer."
    assert result.llm_call_count == 1
    assert limiter.acquire_async.await_count == 1


@pytest.mark.anyio
async def test_generator_prompt_uses_system_a_rules():
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=_response("This is a valid answer."))
    generator = _generator(client)

    await generator.generate("Question?", [_chunk()])

    call = client.chat.completions.create.call_args
    assert call.kwargs["messages"][0] == {"role": "system", "content": GENERATOR_SYSTEM_PROMPT}
    assert "[1] (doc.pdf, p.7, section: \"Section\")" in call.kwargs["messages"][1]["content"]


@pytest.mark.anyio
async def test_generator_prompt_handles_empty_passages():
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=_response("This is a valid answer."))
    generator = _generator(client)

    await generator.generate("Question?", [])

    prompt = client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
    assert "No passages were selected." in prompt


@pytest.mark.anyio
async def test_generator_retries_transient_error():
    limiter = MagicMock(spec=SaiaRateLimiter)
    limiter.acquire_async = AsyncMock()
    client = MagicMock()
    error = InternalServerError("boom", response=MagicMock(status_code=500), body=None)
    client.chat.completions.create = AsyncMock(side_effect=[error, _response("This is a valid answer.")])
    generator = _generator(client, limiter)

    result = await generator.generate("Question?", [_chunk()])

    assert result.llm_call_count == 2
    assert limiter.acquire_async.await_count == 2


@pytest.mark.anyio
async def test_generator_raises_after_retry_budget():
    limiter = MagicMock(spec=SaiaRateLimiter)
    limiter.acquire_async = AsyncMock()
    client = MagicMock()
    error = InternalServerError("boom", response=MagicMock(status_code=500), body=None)
    client.chat.completions.create = AsyncMock(side_effect=error)
    cfg = Settings(system_b=SystemBSettings(generator_retry_attempts=2, generator_retry_wait_seconds=0))
    generator = SystemBGenerator(cfg, limiter, client)

    with pytest.raises(InternalServerError):
        await generator.generate("Question?", [_chunk()])

    assert client.chat.completions.create.call_count == 2


@pytest.mark.anyio
async def test_generator_rejects_empty_content():
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=_response(None))
    generator = _generator(client)

    with pytest.raises(ValueError, match="empty content"):
        await generator.generate("Question?", [_chunk()])


@pytest.mark.anyio
async def test_generator_rejects_short_answer():
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=_response("Too short"))
    generator = _generator(client)

    with pytest.raises(ValueError, match="under 15 characters"):
        await generator.generate("Question?", [_chunk()])
