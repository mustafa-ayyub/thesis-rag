from mcp.server.fastmcp import FastMCP

from src.retrieval.retriever import Retriever
from src.settings import load_settings
from src.types import RetrievedChunk


def chunk_to_payload(chunk: RetrievedChunk) -> dict[str, object]:
    return {
        "chunk_id": chunk.chunk_id,
        "doc_name": chunk.doc_name,
        "doc_title": chunk.doc_title,
        "page_number": chunk.page_number,
        "chunk_type": chunk.chunk_type,
        "section_title": chunk.section_title,
        "chapter": chunk.chapter,
        "token_count": chunk.token_count,
        "text": chunk.text,
        "score": chunk.score,
        "rerank_score": chunk.rerank_score,
    }


def build_server(retriever: Retriever) -> FastMCP:
    server = FastMCP("thesis-retrieval")

    @server.tool()
    def retrieve(query: str) -> list[dict[str, object]]:
        """Search the financial stability report corpus and return the top reranked chunks with metadata."""
        return [chunk_to_payload(chunk) for chunk in retriever.retrieve(query)]

    return server


def main() -> None:
    cfg = load_settings()
    server = build_server(Retriever(cfg))
    server.run("stdio")


if __name__ == "__main__":
    main()
