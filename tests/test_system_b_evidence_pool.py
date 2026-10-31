import pytest

from src.systems.system_b.evidence_pool import EvidencePool, RepeatedQueryError, normalize_query
from src.types import RetrievedChunk


def _chunk(chunk_id: str, text: str = "Passage") -> RetrievedChunk:
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


def test_normalize_query_ignores_case_punctuation_and_spacing():
    assert normalize_query("  NPL   Ratio, 2025? ") == normalize_query("npl ratio 2025")


def test_evidence_pool_assigns_stable_evidence_ids():
    pool = EvidencePool(max_final_chunks=5)
    new_ids, duplicates = pool.add_chunks(1, [_chunk("a"), _chunk("b")])

    assert new_ids == ["E1", "E2"]
    assert duplicates == []
    assert pool.evidence_ids == ["E1", "E2"]


def test_evidence_pool_deduplicates_by_chunk_id():
    pool = EvidencePool(max_final_chunks=5)
    pool.add_chunks(1, [_chunk("a"), _chunk("b")])
    new_ids, duplicates = pool.add_chunks(2, [_chunk("b"), _chunk("c")])

    assert new_ids == ["E3"]
    assert duplicates == ["b"]
    assert pool.evidence_ids == ["E1", "E2", "E3"]
    assert pool.all_chunks == [_chunk("a"), _chunk("b"), _chunk("c")]


def test_evidence_pool_tracks_duplicate_count():
    pool = EvidencePool(max_final_chunks=5)
    pool.add_chunks(1, [_chunk("a")])
    pool.add_chunks(2, [_chunk("a")])
    pool.add_chunks(3, [_chunk("a")])

    assert pool.get_chunk("E1").chunk_id == "a"


def test_evidence_pool_rejects_repeated_query():
    pool = EvidencePool(max_final_chunks=5)
    pool.mark_query("NPL ratio 2025")

    with pytest.raises(RepeatedQueryError):
        pool.mark_query("npl ratio, 2025")

    assert pool.queries == ["NPL ratio 2025"]


def test_evidence_pool_rejects_empty_query():
    pool = EvidencePool(max_final_chunks=5)

    with pytest.raises(ValueError, match="query must not be empty"):
        pool.mark_query("   ")


def test_evidence_pool_selects_chunks_in_requested_order():
    pool = EvidencePool(max_final_chunks=5)
    pool.add_chunks(1, [_chunk("a"), _chunk("b"), _chunk("c")])

    selected = pool.select_chunks(["E3", "E1"])

    assert [chunk.chunk_id for chunk in selected] == ["c", "a"]


def test_evidence_pool_rejects_unknown_evidence_id():
    pool = EvidencePool(max_final_chunks=5)

    with pytest.raises(ValueError, match="Unknown evidence ID"):
        pool.select_chunks(["E9"])


def test_evidence_pool_rejects_duplicate_selection():
    pool = EvidencePool(max_final_chunks=5)
    pool.add_chunks(1, [_chunk("a")])

    with pytest.raises(ValueError, match="unique"):
        pool.select_chunks(["E1", "E1"])


def test_evidence_pool_enforces_final_chunk_limit():
    pool = EvidencePool(max_final_chunks=2)
    pool.add_chunks(1, [_chunk("a"), _chunk("b"), _chunk("c")])

    with pytest.raises(ValueError, match="maximum is 2"):
        pool.select_chunks(["E1", "E2", "E3"])


def test_evidence_pool_renders_empty_state():
    pool = EvidencePool(max_final_chunks=5)

    assert pool.render() == "No evidence has been retrieved yet."


def test_evidence_pool_renders_metadata_and_text():
    pool = EvidencePool(max_final_chunks=5)
    pool.add_chunks(2, [_chunk("a", "Body text")])

    rendered = pool.render()

    assert rendered.startswith("E1 (call 2, doc.pdf, p.1")
    assert "Body text" in rendered
