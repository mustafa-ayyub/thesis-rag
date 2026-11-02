import pytest
from unittest.mock import AsyncMock, MagicMock

from anyio import EndOfStream
from mcp.shared.memory import create_connected_server_and_client_session

from src.saia_rate_limiter import SaiaRateLimiter
from src.settings import SystemBSettings
from src.systems.system_b.mcp_client import (
    McpRetrievalClient,
    McpToolError,
    parse_retrieved_chunks,
)
from src.systems.system_b.mcp_server import build_server, chunk_to_payload
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


def _limiter() -> MagicMock:
    limiter = MagicMock(spec=SaiaRateLimiter)
    limiter.acquire_async = AsyncMock()
    return limiter


def test_chunk_payload_round_trip():
    chunk = _chunk()

    parsed = parse_retrieved_chunks([chunk_to_payload(chunk)])

    assert parsed == [chunk]


def test_parse_retrieved_chunks_accepts_json_string():
    import json

    payload = json.dumps([chunk_to_payload(_chunk())])

    assert parse_retrieved_chunks(payload) == [_chunk()]


def test_parse_retrieved_chunks_unwraps_fastmcp_result_key():
    payload = {"result": [chunk_to_payload(_chunk())]}

    assert parse_retrieved_chunks(payload) == [_chunk()]


def test_parse_retrieved_chunks_rejects_invalid_json():
    with pytest.raises(ValueError, match="not valid JSON"):
        parse_retrieved_chunks("{invalid")


def test_parse_retrieved_chunks_rejects_non_list_payload():
    with pytest.raises(ValueError, match="must be a list"):
        parse_retrieved_chunks({"chunk_id": "abc123"})


@pytest.mark.anyio
async def test_client_retries_transport_error(monkeypatch):
    limiter = _limiter()
    client = McpRetrievalClient(SystemBSettings(mcp_retry_wait_seconds=0), limiter)
    attempts = 0

    async def fake_retrieve_once(query: str) -> list[RetrievedChunk]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise EndOfStream()
        return [_chunk()]

    async def fake_start() -> None:
        return None

    monkeypatch.setattr(client, "_retrieve_once", fake_retrieve_once)
    monkeypatch.setattr(client, "start", fake_start)

    result = await client.retrieve("query")

    assert result == [_chunk()]
    assert attempts == 2
    assert limiter.acquire_async.await_count == 2


@pytest.mark.anyio
async def test_client_retries_tool_error_without_restart(monkeypatch):
    limiter = _limiter()
    client = McpRetrievalClient(SystemBSettings(mcp_retry_wait_seconds=0), limiter)
    attempts = 0
    starts = 0

    async def fake_retrieve_once(query: str) -> list[RetrievedChunk]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise McpToolError("qdrant down")
        return [_chunk()]

    async def fake_start() -> None:
        nonlocal starts
        starts += 1

    monkeypatch.setattr(client, "_retrieve_once", fake_retrieve_once)
    monkeypatch.setattr(client, "start", fake_start)

    result = await client.retrieve("query")

    assert result == [_chunk()]
    assert attempts == 2
    assert starts == 0
    assert limiter.acquire_async.await_count == 2


@pytest.mark.anyio
async def test_client_raises_after_mcp_retry_budget(monkeypatch):
    limiter = _limiter()
    client = McpRetrievalClient(
        SystemBSettings(mcp_retry_attempts=2, mcp_retry_wait_seconds=0), limiter
    )
    attempts = 0

    async def fake_retrieve_once(query: str) -> list[RetrievedChunk]:
        nonlocal attempts
        attempts += 1
        raise EndOfStream()

    async def fake_start() -> None:
        return None

    monkeypatch.setattr(client, "_retrieve_once", fake_retrieve_once)
    monkeypatch.setattr(client, "start", fake_start)

    with pytest.raises(EndOfStream):
        await client.retrieve("query")

    assert attempts == 2
    assert limiter.acquire_async.await_count == 2


class FakeRetriever:
    def __init__(self, chunks: list[RetrievedChunk], error: Exception | None = None):
        self._chunks = chunks
        self._error = error
        self.queries: list[str] = []

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        self.queries.append(query)
        if self._error is not None:
            raise self._error
        return self._chunks


def _extract_chunks(result) -> list[RetrievedChunk]:
    if result.structuredContent is not None:
        return parse_retrieved_chunks(result.structuredContent)
    return parse_retrieved_chunks(result.content[0].text)


@pytest.mark.anyio
async def test_retrieve_tool_round_trips_over_mcp_transport():
    first = _chunk("a")
    second = _chunk("b")
    retriever = FakeRetriever([first, second])
    server = build_server(retriever)

    async with create_connected_server_and_client_session(server) as session:
        tools = await session.list_tools()
        assert [tool.name for tool in tools.tools] == ["retrieve"]

        result = await session.call_tool("retrieve", {"query": "NPL ratio"})

    assert not result.isError
    assert retriever.queries == ["NPL ratio"]
    assert _extract_chunks(result) == [first, second]


@pytest.mark.anyio
async def test_retrieve_tool_surfaces_retriever_error():
    retriever = FakeRetriever([], error=RuntimeError("qdrant down"))
    server = build_server(retriever)

    async with create_connected_server_and_client_session(server) as session:
        result = await session.call_tool("retrieve", {"query": "NPL ratio"})

    assert result.isError
    assert "qdrant down" in _describe(result)


def _describe(result) -> str:
    return " ".join(getattr(item, "text", "") or "" for item in result.content)
