import json
import logging
import os
import sys
from typing import Any

import anyio
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.shared.exceptions import McpError

from src.saia_rate_limiter import SaiaRateLimiter
from src.settings import SystemBSettings
from src.types import RetrievedChunk

logger = logging.getLogger(__name__)

_TRANSPORT_ERRORS = (
    McpError,
    anyio.BrokenResourceError,
    anyio.ClosedResourceError,
    anyio.EndOfStream,
    OSError,
)


def parse_retrieved_chunks(payload: Any) -> list[RetrievedChunk]:
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValueError("MCP retrieval payload is not valid JSON") from exc
    # FastMCP wraps structured tool output under a top-level "result" key
    if isinstance(payload, dict) and "result" in payload:
        payload = payload["result"]
    if not isinstance(payload, list):
        raise ValueError("MCP retrieval payload must be a list")
    return [RetrievedChunk(**item) for item in payload]


class McpToolError(RuntimeError):
    """The MCP server responded but the retrieval tool reported a failure."""


def _describe_content(content: Any) -> str:
    for item in content or []:
        if text := getattr(item, "text", None):
            return text
    return str(content)


class McpRetrievalClient:
    def __init__(
        self,
        settings: SystemBSettings,
        rate_limiter: SaiaRateLimiter,
    ) -> None:
        self._settings = settings
        self._rate_limiter = rate_limiter
        self._params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "src.systems.system_b.mcp_server"],
            env={**os.environ},
        )
        self._streams: Any = None
        self._session: ClientSession | None = None

    async def start(self) -> None:
        self._streams = stdio_client(self._params)
        read, write = await self._streams.__aenter__()
        self._session = ClientSession(read, write)
        await self._session.__aenter__()
        await self._session.initialize()

    async def close(self) -> None:
        if self._session is not None:
            await self._session.__aexit__(None, None, None)
            self._session = None
        if self._streams is not None:
            await self._streams.__aexit__(None, None, None)
            self._streams = None

    async def retrieve(self, query: str) -> list[RetrievedChunk]:
        last_exc: Exception | None = None
        for attempt in range(1, self._settings.mcp_retry_attempts + 1):
            try:
                # The server-side rewriter makes its SAIA call in a separate
                # process without this limiter; a restart resets its cache, so
                # every attempt reserves its own permit.
                await self._rate_limiter.acquire_async()
                return await self._retrieve_once(query)
            except McpToolError as exc:
                last_exc = exc
                logger.warning(
                    "MCP retrieval tool error on attempt %d/%d: %s",
                    attempt,
                    self._settings.mcp_retry_attempts,
                    exc,
                )
            except _TRANSPORT_ERRORS as exc:
                last_exc = exc
                logger.warning(
                    "MCP retrieval transport error on attempt %d/%d: %s",
                    attempt,
                    self._settings.mcp_retry_attempts,
                    exc,
                )
                if attempt < self._settings.mcp_retry_attempts:
                    await self._restart()
            if attempt < self._settings.mcp_retry_attempts:
                await anyio.sleep(self._settings.mcp_retry_wait_seconds)
        assert last_exc is not None
        raise last_exc

    async def _restart(self) -> None:
        await self.close()
        await self.start()

    async def _retrieve_once(self, query: str) -> list[RetrievedChunk]:
        if self._session is None:
            raise RuntimeError("MCP client is not started")

        result = await self._session.call_tool("retrieve", {"query": query})
        if result.isError:
            raise McpToolError(
                f"MCP retrieval tool failed: {_describe_content(result.content)}"
            )

        if result.structuredContent is not None:
            return parse_retrieved_chunks(result.structuredContent)

        if not result.content:
            raise ValueError("MCP retrieval returned no content")

        content = result.content[0]
        text = getattr(content, "text", None)
        if text is None:
            raise ValueError("MCP retrieval returned no text content")

        return parse_retrieved_chunks(text)

    async def __aenter__(self) -> "McpRetrievalClient":
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        await self.close()
