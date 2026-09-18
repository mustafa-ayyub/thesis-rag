import logging

from src.retrieval.remote_post import post_with_retry
from src.settings import Settings
from src.types import RetrievedChunk, ScoredChunk

logger = logging.getLogger(__name__)


class Reranker:
    """
    Wraps BGE-Reranker-v2-m3 for cross-encoder reranking.

    Mode is determined by cfg.gpu.remote_url:
    - Empty string  → load model locally (runs on the GPU cluster)
    - Non-empty URL → forward requests to the FastAPI server via HTTP
                      (laptop mode with the SSH tunnel active)
    """

    def __init__(self, cfg: Settings) -> None:
        self._cfg = cfg
        self._remote_url: str = cfg.gpu.remote_url.rstrip("/")
        self._model = None
        if not self._remote_url:
            self._load_model()

    def _load_model(self) -> None:
        from FlagEmbedding import FlagReranker

        logger.info("Loading %s on %s", self._cfg.models.reranker, self._cfg.gpu.device)
        self._model = FlagReranker(
            self._cfg.models.reranker,
            use_fp16=True,
            device=self._cfg.gpu.device,
        )

    def rerank(
        self, query: str, candidates: list[ScoredChunk], top_n: int
    ) -> list[RetrievedChunk]:
        if self._remote_url:
            return self._rerank_remote(query, candidates, top_n)
        return self._rerank_local(query, candidates, top_n)

    def _rerank_local(
        self, query: str, candidates: list[ScoredChunk], top_n: int
    ) -> list[RetrievedChunk]:
        import torch

        batch_size = self._cfg.gpu.batch_size_rerank
        pairs = [(query, c.text) for c in candidates]
        try:
            scores: list[float] = self._model.compute_score(
                pairs, batch_size=batch_size
            )
        except torch.cuda.OutOfMemoryError:
            logger.warning(
                "OOM at batch_size=%d, retrying at %d", batch_size, batch_size // 2
            )
            torch.cuda.empty_cache()
            scores = self._model.compute_score(pairs, batch_size=batch_size // 2)

        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [
            RetrievedChunk(
                chunk_id=c.chunk_id,
                doc_name=c.doc_name,
                doc_title=c.doc_title,
                page_number=c.page_number,
                chunk_type=c.chunk_type,
                section_title=c.section_title,
                chapter=c.chapter,
                token_count=c.token_count,
                text=c.text,
                score=c.score,
                rerank_score=float(s),
            )
            for c, s in ranked[:top_n]
        ]

    def _rerank_remote(
        self, query: str, candidates: list[ScoredChunk], top_n: int
    ) -> list[RetrievedChunk]:
        payload = {
            "query": query,
            "candidates": [
                {"text": c.text, "index": i} for i, c in enumerate(candidates)
            ],
            "top_n": top_n,
        }
        resp = post_with_retry(
            f"{self._remote_url}/rerank", payload, timeout=60.0
        )
        return [
            RetrievedChunk(
                chunk_id=candidates[item["index"]].chunk_id,
                doc_name=candidates[item["index"]].doc_name,
                doc_title=candidates[item["index"]].doc_title,
                page_number=candidates[item["index"]].page_number,
                chunk_type=candidates[item["index"]].chunk_type,
                section_title=candidates[item["index"]].section_title,
                chapter=candidates[item["index"]].chapter,
                token_count=candidates[item["index"]].token_count,
                text=candidates[item["index"]].text,
                score=candidates[item["index"]].score,
                rerank_score=float(item["score"]),
            )
            for item in resp.json()["results"]
        ]
