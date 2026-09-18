import pytest
from pathlib import Path
from src.settings import load_settings, Settings
from src.ingestion.parser import (
    _clean_lines,
    _is_header_footer_pattern,
    _overlaps_table,
    _extract_text,
    _chart_reason,
)
from src.ingestion.chunker import (
    _chunk_text,
    _table_to_markdown,
    _update_heading_state,
    _extract_chapter,
    _make_chunk_id,
    _count_tokens,
)
import tiktoken


@pytest.fixture
def enc():
    return tiktoken.get_encoding("cl100k_base")


@pytest.fixture
def cfg():
    return load_settings("config/settings.yaml")


def test_settings_load_defaults(cfg):
    assert cfg.retrieval.chunk_size_tokens == 512
    assert cfg.retrieval.chunk_overlap_tokens == 64
    assert cfg.retrieval.hybrid_top_k == 20
    assert cfg.retrieval.rerank_top_n == 5
    assert cfg.qdrant.collection_name == "thesis_corpus"


def test_settings_gpu_remote_url_default(cfg):
    assert cfg.gpu.remote_url == ""


def test_parse_document_missing_file():
    from src.ingestion.parser import parse_document
    with pytest.raises(FileNotFoundError):
        parse_document(Path("data/documents/does_not_exist.pdf"))


def test_chunk_document_empty_pages(cfg):
    from src.ingestion.chunker import chunk_document
    assert chunk_document([], cfg) == []


def test_clean_lines_removes_page_numbers():
    assert _clean_lines("Some text\n42\nMore text") == "Some text\nMore text"


def test_extract_text_filters_footnotes():
    # Page 27 (0-indexed 26) of Bundesbank 2023 has "2 See Barr (2023)." at 8pt —
    # must be absent from extracted text while body prose is preserved.
    import fitz
    path = Path("data/documents/Bundesbank-Financial-Stability-Review-2023.pdf")
    if not path.exists():
        pytest.skip("PDF not in repo")
    doc = fitz.open(str(path))
    text = _extract_text(doc[26], [])
    assert "See Barr" not in text
    assert len(text) > 50  # body text still present


def test_clean_lines_keeps_numbered_section_headings():
    result = _clean_lines("Intro text\n2 Financial markets\nMore prose")
    assert "2 Financial markets" in result


def test_clean_lines_removes_note_and_source_labels():
    result = _clean_lines("Normal\nNote: ECB data.\nSource: Bundesbank.\nMore")
    assert "Note:" not in result and "Source:" not in result
    assert "Normal" in result and "More" in result


def test_clean_lines_keeps_regular_content():
    text = "Financial stability risks increased.\nBank leverage remained elevated."
    assert _clean_lines(text) == text


def test_header_footer_pure_digit():
    assert _is_header_footer_pattern("42") is True


def test_header_footer_date_string():
    assert _is_header_footer_pattern("12/2023") is True


def test_header_footer_very_short():
    assert _is_header_footer_pattern("ECB") is True


def test_header_footer_normal_title():
    assert _is_header_footer_pattern("Financial Stability Review") is False


def test_overlaps_table_full_overlap():
    assert _overlaps_table((10.0, 10.0, 100.0, 100.0), [(10.0, 10.0, 100.0, 100.0)]) is True


def test_overlaps_table_no_overlap():
    assert _overlaps_table((10.0, 10.0, 50.0, 50.0), [(200.0, 200.0, 300.0, 300.0)]) is False


def test_overlaps_table_small_overlap_below_threshold():
    # 10×10 overlap on a 100×100 block = 1% < 30% threshold
    assert _overlaps_table((0.0, 0.0, 100.0, 100.0), [(90.0, 90.0, 200.0, 200.0)]) is False


def test_overlaps_table_empty_list():
    assert _overlaps_table((0.0, 0.0, 100.0, 100.0), []) is False


def test_chart_reason_chart_reference_in_header():
    table = [
        ["Distribution of interest rate changes Chart 2.1.1", ""],
        ["30\n25\n20", "10\n5\n0"],
        ["Deutsche Bundesbank", ""],
    ]
    assert _chart_reason(table, "") is not None


def test_chart_reason_chart_reference_in_title():
    table = [["Real estate prices Chart 2.1.5", ""], ["180\n160\n140", "100\n80"]]
    assert _chart_reason(table, "") is not None
    # title-only reference (chart caption above the box)
    assert _chart_reason([["x", "y"], ["a", "b"]], "Real estate prices Chart 2.1.5") is not None


def test_chart_reason_numeric_lines():
    # Mostly bare axis values, a few label lines (SVB metrics chart)
    table = [
        ["Selected metrics for SVB Financial Group", ""],
        ["US$ billion\nSecurities portfolio\n200\n150\n100\n50\n0", "Interest income\n2.3\n2.0\n1.7\n1.4\n2020 2021 2022"],
        ["Source: SVB quarterly reports.", ""],
    ]
    assert _chart_reason(table, "") is not None


def test_chart_reason_axis_ticks():
    table = [
        ["Interest rate fixation periods", ""],
        ["% \n100\n90\n80\n70", "200405060708091011121314151617181920212223"],
        ["Deutsche Bundesbank", ""],
    ]
    assert _chart_reason(table, "") is not None


def test_chart_reason_real_table_kept():
    table = [
        ["NGFS scenario vintage", "2021 analysis", "2023 analysis"],
        ["Analytical goals", "Estimate losses in portfolios", "Estimate losses and gauge resilience"],
        ["Application of climate scenarios", "Banking sector only", "Banking and insurance sectors"],
    ]
    assert _chart_reason(table, "Main changes to the analytical framework Table 3.1") is None


def test_chart_reason_small_real_table_with_numbers_kept():
    # Meta-analysis table: numeric cells but text in every row
    table = [
        ["Outcome variable", "Sample size", "Uncorrected effect", "Evidence for bias", "Bias-corrected effect"],
        ["Housing credit growth", "1,140", "-0.0005", "Very strong", "-0.0369***"],
        ["", "927", "-0.0098", "Strong", ""],
    ]
    assert _chart_reason(table, "") is None


def test_chunk_text_short_text_single_chunk(enc):
    # Text must be >= MIN_CHUNK_TOKENS (50) to survive the discard filter
    sentence = "The European Central Bank monitors systemic financial stability risks. "
    text = (sentence * 5).strip()  # ~60 tokens, well above threshold
    chunks = _chunk_text(text, max_tokens=512, overlap_tokens=64, enc=enc)
    assert len(chunks) == 1


def test_chunk_text_respects_max_tokens(enc):
    # Generate text that is definitely > 512 tokens
    sentence = "The European Central Bank monitors financial stability risks. "
    long_text = sentence * 20
    chunks = _chunk_text(long_text, max_tokens=512, overlap_tokens=64, enc=enc)
    for chunk in chunks:
        assert _count_tokens(chunk, enc) <= 512


def test_chunk_text_many_small_atoms_stay_under_max(enc):
    # Many short atoms packed into one chunk: the "\n" separators between them
    # must be counted or the flushed chunk overshoots max_tokens.
    sentence = "Risk rose. "  # ~3 tokens each → separators dominate
    long_text = sentence * 60
    chunks = _chunk_text(long_text, max_tokens=40, overlap_tokens=0, enc=enc)
    for chunk in chunks:
        assert _count_tokens(chunk, enc) <= 40


def test_chunk_text_discards_short_fragments(enc):
    # Very short text that is < MIN_CHUNK_TOKENS (50) should produce no chunks
    text = "Short."
    chunks = _chunk_text(text, max_tokens=512, overlap_tokens=64, enc=enc)
    assert chunks == []


def test_chunk_text_overlap_between_chunks(enc):
    # 15 sentences × ~13 tokens = ~195 tokens; max_tokens=100 forces at least 2 chunks
    sentence = "Financial stability risks in the euro area remained elevated during this period. "
    long_text = sentence * 15
    chunks = _chunk_text(long_text, max_tokens=100, overlap_tokens=16, enc=enc)
    assert len(chunks) >= 2
    # Consecutive chunks should share some tokens (overlap)
    if len(chunks) >= 2:
        # Last ~32 tokens of chunk 0 should appear somewhere in chunk 1
        end_of_first = chunks[0][-80:]
        assert any(word in chunks[1] for word in end_of_first.split()[-4:])


def test_chunk_text_empty_input(enc):
    assert _chunk_text("", max_tokens=512, overlap_tokens=64, enc=enc) == []


def test_table_to_markdown_basic():
    table = [["Country", "GDP", "Debt"], ["Germany", "4.1", "65%"], ["France", "3.0", "112%"]]
    md = _table_to_markdown(table, page_num=1, doc_name="test.pdf")
    assert md is not None
    assert "| Country | GDP | Debt |" in md
    assert "| --- | --- | --- |" in md
    assert "| Germany | 4.1 | 65% |" in md


def test_table_to_markdown_pads_short_rows():
    table = [["A", "B", "C"], ["x", "y"]]  # row 1 has only 2 cells
    md = _table_to_markdown(table, page_num=1, doc_name="test.pdf")
    assert md is not None
    assert "| x | y |  |" in md


def test_update_heading_state_detects_all_caps():
    title, chapter = _update_heading_state(
        "Some intro text\nFINANCIAL STABILITY OVERVIEW\nMore text", "", ""
    )
    assert title == "FINANCIAL STABILITY OVERVIEW"


def test_update_heading_state_detects_numbered_section():
    title, chapter = _update_heading_state(
        "Preamble\n2.1 Financial markets under stress\nDetails", "", ""
    )
    assert title == "2.1 Financial markets under stress"
    assert chapter.startswith("2.1")


def test_update_heading_state_no_heading_preserves_state():
    title, chapter = _update_heading_state(
        "No heading here, just prose text.", "Previous Section", "1 Previous"
    )
    assert title == "Previous Section"
    assert chapter == "1 Previous"


def test_extract_chapter_simple():
    assert _extract_chapter("2 Financial markets") == "2 Financial markets"


def test_extract_chapter_subsection():
    result = _extract_chapter("2.1 Financial markets have been very volatile this year")
    assert result.startswith("2.1")
    words = result.split()
    assert len(words) <= 7  # prefix + up to 6 words


def test_extract_chapter_non_numbered():
    assert _extract_chapter("FINANCIAL STABILITY OVERVIEW") == ""


def test_make_chunk_id_is_16_chars():
    cid = _make_chunk_id("doc.pdf", 5, "some text")
    assert len(cid) == 16


def test_make_chunk_id_deterministic():
    a = _make_chunk_id("doc.pdf", 5, "some text")
    b = _make_chunk_id("doc.pdf", 5, "some text")
    assert a == b


def test_make_chunk_id_differs_by_page():
    a = _make_chunk_id("doc.pdf", 1, "same text")
    b = _make_chunk_id("doc.pdf", 2, "same text")
    assert a != b


def test_make_chunk_id_differs_when_only_tail_differs():
    # Two tables sharing the same first 80 chars but differing later must not
    # collide (else Qdrant upsert silently overwrites the first).
    prefix = "Table 3.1 Bank capital ratios by country and year in the euro area "
    a = _make_chunk_id("doc.pdf", 5, prefix + "Germany 17%")
    b = _make_chunk_id("doc.pdf", 5, prefix + "France 15%")
    assert a != b
