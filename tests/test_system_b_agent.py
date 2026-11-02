from unittest.mock import AsyncMock, MagicMock

import pytest
from openai import InternalServerError

from src.saia_rate_limiter import SaiaRateLimiter
from src.settings import Settings, SystemBSettings
from src.systems.system_b.agent import AGENT_SYSTEM_PROMPT, SystemBAgent, parse_decision
from src.systems.system_b.evidence_pool import EvidencePool
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
        text="A retrieved passage.",
        score=0.5,
        rerank_score=0.9,
    )


def _response(content: str, finish_reason: str = "stop") -> MagicMock:
    return MagicMock(
        choices=[MagicMock(finish_reason=finish_reason, message=MagicMock(content=content))]
    )


def _agent(
    client: MagicMock,
    limiter: MagicMock | None = None,
    cfg: Settings | None = None,
) -> SystemBAgent:
    return SystemBAgent(
        cfg or Settings(system_b=SystemBSettings(agent_retry_base_seconds=0)),
        limiter or MagicMock(spec=SaiaRateLimiter),
        client,
    )


def _pool() -> EvidencePool:
    pool = EvidencePool(max_final_chunks=5)
    pool.add_chunks(1, [_chunk()])
    return pool


def test_parse_decision_accepts_json_object():
    decision = parse_decision('{"decision": "stop", "selected_evidence_ids": ["E1"]}')

    assert decision.decision == "stop"
    assert decision.selected_evidence_ids == ["E1"]


def test_parse_decision_accepts_json_code_fence():
    content = '```json\n{"decision": "stop"}\n```'

    assert parse_decision(content).decision == "stop"


def test_parse_decision_rejects_missing_json():
    with pytest.raises(ValueError, match="JSON object"):
        parse_decision("no json here")


@pytest.mark.anyio
async def test_agent_decide_returns_decision_and_counts_call():
    limiter = MagicMock(spec=SaiaRateLimiter)
    limiter.acquire_async = AsyncMock()
    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        return_value=_response('{"decision": "stop", "selected_evidence_ids": ["E1"]}')
    )
    agent = _agent(client, limiter)

    decision, llm_calls = await agent.decide("question?", _pool(), 1)

    assert decision.decision == "stop"
    assert llm_calls == 1
    assert limiter.acquire_async.await_count == 1
    call = client.chat.completions.create.call_args
    assert call.kwargs["model"] == "qwen3.8-27b"
    assert call.kwargs["temperature"] == 0
    assert call.kwargs["max_tokens"] == 800
    assert call.kwargs["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert call.kwargs["messages"][0] == {"role": "system", "content": AGENT_SYSTEM_PROMPT}


@pytest.mark.anyio
async def test_agent_prompt_contains_state_and_feedback():
    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        return_value=_response('{"decision": "stop"}')
    )
    agent = _agent(client)
    pool = _pool()
    pool.mark_query("previous query")

    await agent.decide("question?", pool, 1, feedback="do not repeat")

    prompt = client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
    assert "Question: question?" in prompt
    assert "Retrieval calls used: 1 / 3" in prompt
    assert "- previous query" in prompt
    assert "E1" in prompt
    assert "Maximum final chunks: 5" in prompt
    assert "Feedback: do not repeat" in prompt


@pytest.mark.anyio
async def test_agent_retries_invalid_json():
    limiter = MagicMock(spec=SaiaRateLimiter)
    limiter.acquire_async = AsyncMock()
    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        side_effect=[
            _response("not json"),
            _response('{"decision": "stop"}'),
        ]
    )
    agent = _agent(client, limiter)

    decision, llm_calls = await agent.decide("question?", _pool(), 0)

    assert decision.decision == "stop"
    assert llm_calls == 2
    assert limiter.acquire_async.await_count == 2


@pytest.mark.anyio
async def test_agent_retries_transient_error():
    limiter = MagicMock(spec=SaiaRateLimiter)
    limiter.acquire_async = AsyncMock()
    client = MagicMock()
    error = InternalServerError("boom", response=MagicMock(status_code=500), body=None)
    client.chat.completions.create = AsyncMock(
        side_effect=[error, _response('{"decision": "stop"}')]
    )
    agent = _agent(client, limiter)

    decision, llm_calls = await agent.decide("question?", _pool(), 0)

    assert decision.decision == "stop"
    assert llm_calls == 2


@pytest.mark.anyio
async def test_agent_raises_after_retry_budget():
    limiter = MagicMock(spec=SaiaRateLimiter)
    limiter.acquire_async = AsyncMock()
    client = MagicMock()
    error = InternalServerError("boom", response=MagicMock(status_code=500), body=None)
    client.chat.completions.create = AsyncMock(side_effect=error)
    cfg = Settings(system_b=SystemBSettings(agent_retry_attempts=2, agent_retry_base_seconds=0))
    agent = SystemBAgent(cfg, limiter, client)

    with pytest.raises(InternalServerError):
        await agent.decide("question?", _pool(), 0)

    assert client.chat.completions.create.call_count == 2
