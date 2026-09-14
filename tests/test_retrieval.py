import pytest
from unittest.mock import MagicMock, patch
from src.settings import Settings
from src.retrieval.hybrid_search import HybridSearch
from src.types import ScoredChunk


def _fake_hit(chunk_id: str, doc_name: str = "doc.pdf", page: int = 1) -> MagicMock:
    hit = MagicMock()
    hit.payload = {
        "chunk_id": chunk_id,
        "doc_name": doc_name,
        "doc_title": "Test Doc",
        "page_number": page,
        "chunk_type": "text",
        "section_title": "",
        "chapter": "",
        "token_count": 80,
        "text": f"text for {chunk_id}",
    }
    return hit


def _make_search(remote_url: str = "http://localhost:8000") -> HybridSearch:
    cfg = Settings()
    cfg.gpu.remote_url = remote_url
    return HybridSearch(cfg)


def test_retriever_imports():
    from src.retrieval.retriever import Retriever
    assert Retriever is not None


def test_embedder_remote_mode_does_not_load_model():
    from src.settings import Settings
    from src.retrieval.embedder import Embedder

    cfg = Settings()
    cfg.gpu.remote_url = "http://localhost:8000"
    embedder = Embedder(cfg)
    assert embedder._model is None


def test_reranker_remote_mode_does_not_load_model():
    from src.settings import Settings
    from src.retrieval.reranker import Reranker

    cfg = Settings()
    cfg.gpu.remote_url = "http://localhost:8000"
    reranker = Reranker(cfg)
    assert reranker._model is None


def test_rrf_merges_disjoint_lists():
    hs = _make_search()
    dense_hits = [_fake_hit("a"), _fake_hit("b")]
    sparse_hits = [_fake_hit("c"), _fake_hit("d")]
    results = hs._rrf(dense_hits, sparse_hits, top_k=4)
    ids = {r.chunk_id for r in results}
    assert ids == {"a", "b", "c", "d"}


def test_rrf_boosts_overlapping_chunk():
    hs = _make_search()
    # "shared" appears rank-1 in dense and rank-1 in sparse → highest RRF score
    dense_hits = [_fake_hit("shared"), _fake_hit("dense_only")]
    sparse_hits = [_fake_hit("shared"), _fake_hit("sparse_only")]
    results = hs._rrf(dense_hits, sparse_hits, top_k=3)
    assert results[0].chunk_id == "shared"


def test_rrf_respects_top_k():
    hs = _make_search()
    dense_hits = [_fake_hit(f"d{i}") for i in range(5)]
    sparse_hits = [_fake_hit(f"s{i}") for i in range(5)]
    results = hs._rrf(dense_hits, sparse_hits, top_k=3)
    assert len(results) == 3


def test_rrf_returns_scored_chunks():
    hs = _make_search()
    dense_hits = [_fake_hit("x")]
    results = hs._rrf(dense_hits, [], top_k=5)
    assert len(results) == 1
    assert isinstance(results[0], ScoredChunk)
    assert results[0].score > 0


def test_rrf_score_formula():
    hs = _make_search()
    rrf_k = hs._cfg.retrieval.rrf_k  # 60
    dense_hits = [_fake_hit("a")]
    sparse_hits = [_fake_hit("a")]
    results = hs._rrf(dense_hits, sparse_hits, top_k=1)
    expected = 2.0 / (rrf_k + 1)
    assert abs(results[0].score - expected) < 1e-9


def test_query_rewriter_caches_result():
    from src.retrieval.query_rewriter import QueryRewriter

    cfg = Settings()
    with patch("src.retrieval.query_rewriter.OpenAI") as MockOpenAI:
        mock_client = MagicMock()
        MockOpenAI.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="rewritten query"))]
        )
        qr = QueryRewriter(cfg)
        first = qr.rewrite("what are financial risks?")
        second = qr.rewrite("what are financial risks?")

    assert first == second == "rewritten query"
    mock_client.chat.completions.create.assert_called_once()


def test_query_rewriter_different_queries_not_cached():
    from src.retrieval.query_rewriter import QueryRewriter

    cfg = Settings()
    with patch("src.retrieval.query_rewriter.OpenAI") as MockOpenAI:
        mock_client = MagicMock()
        MockOpenAI.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="result"))]
        )
        qr = QueryRewriter(cfg)
        qr.rewrite("query one")
        qr.rewrite("query two")

    assert mock_client.chat.completions.create.call_count == 2


def test_query_rewriter_retries_on_server_error():
    from openai import InternalServerError
    from src.retrieval.query_rewriter import QueryRewriter

    cfg = Settings()
    with patch("src.retrieval.query_rewriter.OpenAI") as MockOpenAI, patch(
        "src.retrieval.query_rewriter.time.sleep"
    ) as mock_sleep:
        mock_client = MagicMock()
        MockOpenAI.return_value = mock_client
        mock_client.chat.completions.create.side_effect = [
            InternalServerError("boom", response=MagicMock(status_code=500), body=None),
            MagicMock(choices=[MagicMock(message=MagicMock(content="rewritten"))]),
        ]
        qr = QueryRewriter(cfg)
        result = qr.rewrite("test query")

    assert result == "rewritten"
    assert mock_client.chat.completions.create.call_count == 2
    mock_sleep.assert_called_once_with(30.0)


def test_query_rewriter_gives_up_after_max_attempts():
    from openai import InternalServerError
    from src.retrieval.query_rewriter import QueryRewriter

    cfg = Settings()
    err = InternalServerError("boom", response=MagicMock(status_code=500), body=None)
    with patch("src.retrieval.query_rewriter.OpenAI") as MockOpenAI, patch(
        "src.retrieval.query_rewriter.time.sleep"
    ) as mock_sleep:
        mock_client = MagicMock()
        MockOpenAI.return_value = mock_client
        mock_client.chat.completions.create.side_effect = err
        qr = QueryRewriter(cfg)
        with pytest.raises(InternalServerError):
            qr.rewrite("test query")

    assert mock_client.chat.completions.create.call_count == 5
    assert mock_sleep.call_count == 4
