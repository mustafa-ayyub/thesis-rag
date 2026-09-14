import logging

from src.retrieval.embedder import Embedder
from src.retrieval.hybrid_search import HybridSearch
from src.retrieval.query_rewriter import QueryRewriter
from src.retrieval.reranker import Reranker
from src.settings import Settings
from src.types import RetrievedChunk

logger = logging.getLogger(__name__)


class Retriever:
    def __init__(self, cfg: Settings) -> None:
        self._cfg = cfg
        self._rewriter = QueryRewriter(cfg)
        self._embedder = Embedder(cfg)
        self._search = HybridSearch(cfg)
        self._reranker = Reranker(cfg)

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        rewritten = self._rewriter.rewrite(query)
        dense, sparse = self._embedder.embed_query(rewritten)
        candidates = self._search.search(
            dense, sparse, top_k=self._cfg.retrieval.hybrid_top_k
        )
        return self._reranker.rerank(
            rewritten, candidates, top_n=self._cfg.retrieval.rerank_top_n
        )
