from dataclasses import dataclass


@dataclass
class ChunkMetadata:
    chunk_id: str       # sha256[:16] of f"{doc_name}|{page_number}|{text}"
    doc_name: str
    doc_title: str
    page_number: int
    chunk_type: str     # "text" | "table"
    section_title: str
    chapter: str
    token_count: int
    text: str


@dataclass
class ScoredChunk:
    """Candidate returned by hybrid search, before reranking."""
    chunk_id: str
    doc_name: str
    doc_title: str
    page_number: int
    chunk_type: str
    section_title: str
    chapter: str
    token_count: int
    text: str
    score: float        # RRF hybrid score


@dataclass
class RetrievedChunk:
    """Final result returned by the Retriever after reranking."""
    chunk_id: str
    doc_name: str
    doc_title: str
    page_number: int
    chunk_type: str
    section_title: str
    chapter: str
    token_count: int
    text: str
    score: float        # RRF hybrid score
    rerank_score: float # cross-encoder score
