from unittest.mock import AsyncMock, MagicMock

import pytest

from src.saia_rate_limiter import SaiaRateLimiter
from src.settings import Settings, SystemBSettings
from src.systems.system_b.pipeline import SystemBPipeline
from src.systems.system_b.types import (
    AgentDecision,
    GenerationResult,
)
from src.types import RetrievedChunk


def _chunk(chunk_id: str, text: str = "A retrieved passage.") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_name="doc.pdf",
        doc_title="Test document",
        page_number=1,
        chunk_type="text",
        section_title="Section",
        chapter="1 Chapter",
        token_count=8,
        text=text,
        score=0.5,
        rerank_score=0.9,
    )


def _settings(**kwargs) -> Settings:
    return Settings(system_b=SystemBSettings(**kwargs))


def _limiter() -> MagicMock:
    limiter = MagicMock(spec=SaiaRateLimiter)
    limiter.acquire_async = AsyncMock()
    return limiter


class FakeRetrievalTool:
    def __init__(self, responses: dict[str, list[RetrievedChunk]]) -> None:
        self._responses = responses
        self.queries: list[str] = []
        self.started = False
        self.closed = False

    async def start(self) -> None:
        self.started = True

    async def close(self) -> None:
        self.closed = True

    async def retrieve(self, query: str) -> list[RetrievedChunk]:
        self.queries.append(query)
        return self._responses[query]


class FakeAgent:
    def __init__(self, decisions: list[dict], llm_calls: int = 1) -> None:
        self._decisions = [AgentDecision.model_validate(d) for d in decisions]
        self._llm_calls = llm_calls
        self.call_count = 0
        self.feedback_log: list[str | None] = []

    async def decide(
        self,
        question: str,
        pool,
        retrieval_calls_used: int,
        feedback: str | None,
    ) -> tuple[AgentDecision, int]:
        self.call_count += 1
        self.feedback_log.append(feedback)
        return self._decisions.pop(0), self._llm_calls


class FakeGenerator:
    def __init__(self, answer: str = "The answer is based on the evidence.") -> None:
        self.answer = answer
        self.passages_log: list[list[RetrievedChunk]] = []

    async def generate(self, question: str, passages: list[RetrievedChunk]) -> GenerationResult:
        self.passages_log.append(passages)
        return GenerationResult(answer=self.answer, llm_call_count=1)


def _pipeline(
    tool: FakeRetrievalTool,
    agent: FakeAgent,
    generator: FakeGenerator,
    settings: Settings | None = None,
    limiter: MagicMock | None = None,
) -> SystemBPipeline:
    return SystemBPipeline(
        settings or _settings(),
        limiter or _limiter(),
        mcp_client=tool,
        agent=agent,
        generator=generator,
    )


@pytest.mark.anyio
async def test_run_happy_path_retrieve_then_stop():
    chunks = [_chunk("a"), _chunk("b")]
    tool = FakeRetrievalTool({"NPL ratio 2025": chunks})
    agent = FakeAgent(
        [
            {"decision": "retrieve_more", "next_query": "NPL ratio 2025"},
            {"decision": "stop", "selected_evidence_ids": ["E2", "E1"]},
        ]
    )
    generator = FakeGenerator()
    limiter = _limiter()
    pipeline = _pipeline(tool, agent, generator, limiter=limiter)

    result = await pipeline.run("What is the NPL ratio?")

    assert result.answer == generator.answer
    assert [c.chunk_id for c in result.final_chunks] == ["b", "a"]
    assert [c.chunk_id for c in result.all_chunks] == ["a", "b"]
    assert result.retrieval_call_count == 1
    assert result.llm_call_count == 3
    assert result.warnings == []
    assert tool.queries == ["NPL ratio 2025"]
    assert generator.passages_log == [[chunks[1], chunks[0]]]
    call = result.retrieval_calls[0]
    assert call.call_number == 1
    assert call.query == "NPL ratio 2025"
    assert call.new_evidence_ids == ["E1", "E2"]
    assert call.duplicate_chunk_ids == []
    assert isinstance(call.latency_ms, int)
    assert isinstance(result.latency_ms, int)


@pytest.mark.anyio
async def test_run_enforces_max_retrieval_calls():
    tool = FakeRetrievalTool({f"query {i}": [_chunk(f"c{i}")] for i in range(1, 5)})
    agent = FakeAgent(
        [{"decision": "retrieve_more", "next_query": f"query {i}"} for i in range(1, 5)]
    )
    generator = FakeGenerator()
    pipeline = _pipeline(tool, agent, generator)

    result = await pipeline.run("Question?")

    assert result.retrieval_call_count == 3
    assert tool.queries == ["query 1", "query 2", "query 3"]
    assert agent.call_count == 4
    assert [c.chunk_id for c in result.final_chunks] == ["c1", "c2", "c3"]
    assert any("maximum number of calls" in warning for warning in result.warnings)


@pytest.mark.anyio
async def test_run_repeated_query_sends_feedback_and_continues():
    chunks = [_chunk("a")]
    tool = FakeRetrievalTool({"NPL ratio": chunks})
    agent = FakeAgent(
        [
            {"decision": "retrieve_more", "next_query": "NPL ratio"},
            {"decision": "retrieve_more", "next_query": "npl, ratio"},
            {"decision": "stop", "selected_evidence_ids": ["E1"]},
        ]
    )
    generator = FakeGenerator()
    pipeline = _pipeline(tool, agent, generator)

    result = await pipeline.run("Question?")

    assert result.retrieval_call_count == 1
    assert tool.queries == ["NPL ratio"]
    assert agent.call_count == 3
    assert agent.feedback_log[0] is None
    assert agent.feedback_log[1] is None
    assert agent.feedback_log[2] is not None
    assert "already used" in agent.feedback_log[2]
    assert any("repeated query" in warning for warning in result.warnings)
    assert [c.chunk_id for c in result.final_chunks] == ["a"]


@pytest.mark.anyio
async def test_run_persistent_unusable_query_finalizes_without_retrieval():
    tool = FakeRetrievalTool({})
    agent = FakeAgent(
        [{"decision": "retrieve_more", "next_query": "??!"} for _ in range(3)]
    )
    generator = FakeGenerator()
    pipeline = _pipeline(tool, agent, generator)

    result = await pipeline.run("Question?")

    assert result.retrieval_call_count == 0
    assert tool.queries == []
    assert agent.call_count == 3
    assert result.final_chunks == []
    assert generator.passages_log == [[]]
    assert any("persisted" in warning for warning in result.warnings)
    assert any("no searchable words" in feedback for feedback in agent.feedback_log[1:])


@pytest.mark.anyio
async def test_run_stops_immediately_with_empty_pool():
    tool = FakeRetrievalTool({})
    agent = FakeAgent([{"decision": "stop", "selected_evidence_ids": []}])
    generator = FakeGenerator()
    pipeline = _pipeline(tool, agent, generator)

    result = await pipeline.run("Question not answerable from the corpus?")

    assert result.retrieval_call_count == 0
    assert result.final_chunks == []
    assert result.llm_call_count == 2
    assert result.warnings == []
    assert generator.passages_log == [[]]


@pytest.mark.anyio
async def test_run_empty_selection_with_evidence_falls_back():
    chunks = [_chunk("a"), _chunk("b")]
    tool = FakeRetrievalTool({"query 1": chunks})
    agent = FakeAgent(
        [
            {"decision": "retrieve_more", "next_query": "query 1"},
            {"decision": "stop", "selected_evidence_ids": []},
        ]
    )
    generator = FakeGenerator()
    pipeline = _pipeline(tool, agent, generator)

    result = await pipeline.run("Question?")

    assert [c.chunk_id for c in result.final_chunks] == ["a", "b"]
    assert any("without selecting evidence" in warning for warning in result.warnings)
    assert any("Falling back" in warning for warning in result.warnings)


@pytest.mark.anyio
async def test_run_invalid_selection_falls_back():
    tool = FakeRetrievalTool({"query 1": [_chunk("a")]})
    agent = FakeAgent(
        [
            {"decision": "retrieve_more", "next_query": "query 1"},
            {"decision": "stop", "selected_evidence_ids": ["E9"]},
        ]
    )
    generator = FakeGenerator()
    pipeline = _pipeline(tool, agent, generator)

    result = await pipeline.run("Question?")

    assert [c.chunk_id for c in result.final_chunks] == ["a"]
    assert any("Invalid agent selection" in warning for warning in result.warnings)


@pytest.mark.anyio
async def test_run_tracks_duplicates_across_calls():
    a, b, c = _chunk("a"), _chunk("b"), _chunk("c")
    tool = FakeRetrievalTool({"query 1": [a, b], "query 2": [b, c]})
    agent = FakeAgent(
        [
            {"decision": "retrieve_more", "next_query": "query 1"},
            {"decision": "retrieve_more", "next_query": "query 2"},
            {"decision": "stop", "selected_evidence_ids": ["E3"]},
        ]
    )
    generator = FakeGenerator()
    pipeline = _pipeline(tool, agent, generator)

    result = await pipeline.run("Question?")

    assert result.retrieval_calls[0].new_evidence_ids == ["E1", "E2"]
    assert result.retrieval_calls[1].new_evidence_ids == ["E3"]
    assert result.retrieval_calls[1].duplicate_chunk_ids == ["b"]
    assert [c2.chunk_id for c2 in result.all_chunks] == ["a", "b", "c"]
    assert [c2.chunk_id for c2 in result.final_chunks] == ["c"]


def test_pipeline_default_client_shares_rate_limiter():
    limiter = _limiter()
    pipeline = SystemBPipeline(
        _settings(),
        limiter,
        agent=FakeAgent([{"decision": "stop", "selected_evidence_ids": []}]),
        generator=FakeGenerator(),
    )

    assert pipeline._mcp_client._rate_limiter is limiter


@pytest.mark.anyio
async def test_run_accumulates_llm_call_count_across_retries():
    tool = FakeRetrievalTool({})
    agent = FakeAgent([{"decision": "stop", "selected_evidence_ids": []}], llm_calls=2)
    generator = FakeGenerator()
    pipeline = _pipeline(tool, agent, generator)

    result = await pipeline.run("Question?")

    assert result.llm_call_count == 3


@pytest.mark.anyio
async def test_context_manager_starts_and_closes_retrieval_tool():
    tool = FakeRetrievalTool({})
    agent = FakeAgent([{"decision": "stop", "selected_evidence_ids": []}])
    generator = FakeGenerator()
    pipeline = _pipeline(tool, agent, generator)

    async with pipeline:
        result = await pipeline.run("Question?")

    assert result.answer == generator.answer
    assert tool.started
    assert tool.closed
