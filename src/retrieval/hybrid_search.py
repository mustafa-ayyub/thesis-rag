"""Hybrid Qdrant search with Reciprocal Rank Fusion.

Dense and sparse searches run independently; results are merged via RRF.
"""
import logging
from typing import TYPE_CHECKING

from src.settings import Settings
from src.types import ScoredChunk

if TYPE_CHECKING:
    from qdrant_client import QdrantClient
    from qdrant_client.models import ScoredPoint

logger = logging.getLogger(__name__)


class HybridSearch:
    """Queries Qdrant with dense + sparse vectors and merges via RRF."""

    def __init__(self, cfg: Settings) -> None:
        self._cfg = cfg
        self._client: "QdrantClient | None" = None

    def _get_client(self) -> "QdrantClient":
        if self._client is None:
            from qdrant_client import QdrantClient
            self._client = QdrantClient(
                host=self._cfg.qdrant.host,
                port=self._cfg.qdrant.port,
            )
        return self._client

    def search(
        self,
        dense: list[float],
        sparse: dict[int, float],
        top_k: int,
    ) -> list[ScoredChunk]:
        """Search Qdrant and merge results with Reciprocal Rank Fusion."""
        from qdrant_client.models import SparseVector

        client = self._get_client()
        collection = self._cfg.qdrant.collection_name

        dense_hits = client.query_points(
            collection_name=collection,
            query=dense,
            using=self._cfg.qdrant.dense_vector_name,
            limit=top_k,
            with_payload=True,
        ).points

        sparse_hits = client.query_points(
            collection_name=collection,
            query=SparseVector(
                indices=list(sparse.keys()),
                values=list(sparse.values()),
            ),
            using=self._cfg.qdrant.sparse_vector_name,
            limit=top_k,
            with_payload=True,
        ).points

        results = self._rrf(dense_hits, sparse_hits, top_k)
        logger.debug(
            "Hybrid search: %d dense + %d sparse → %d merged",
            len(dense_hits), len(sparse_hits), len(results),
        )
        return results

    def _rrf(
        self,
        dense_hits: list["ScoredPoint"],
        sparse_hits: list["ScoredPoint"],
        top_k: int,
    ) -> list[ScoredChunk]:
        rrf_k = self._cfg.retrieval.rrf_k
        scores: dict[str, float] = {}
        payloads: dict[str, dict] = {}

        for rank, hit in enumerate(dense_hits, start=1):
            cid = hit.payload["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)
            payloads[cid] = hit.payload

        for rank, hit in enumerate(sparse_hits, start=1):
            cid = hit.payload["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)
            if cid not in payloads:
                payloads[cid] = hit.payload

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        return [
            ScoredChunk(
                chunk_id=cid,
                doc_name=p["doc_name"],
                doc_title=p["doc_title"],
                page_number=p["page_number"],
                chunk_type=p["chunk_type"],
                section_title=p["section_title"],
                chapter=p["chapter"],
                token_count=p["token_count"],
                text=p["text"],
                score=score,
            )
            for cid, score in ranked
            for p in (payloads[cid],)
        ]
