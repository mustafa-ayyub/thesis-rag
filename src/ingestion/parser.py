"""PDF parser: extracts per-page text and tables from FSR documents.

Uses pdfplumber for table detection and pymupdf for text extraction.
Both libraries use a top-left origin coordinate system (y increases downward).
"""
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import fitz  # pymupdf
import pdfplumber
import pdfplumber.page

logger = logging.getLogger(__name__)

# Top and bottom fraction of page height treated as header/footer zone
_MARGIN_RATIO = 0.07
# Vertical range (points) to search above a table for its title
_TITLE_SEARCH_GAP = 35.0
# Overlap fraction of a text block's area required to classify it as "inside a table"
_TABLE_OVERLAP_THRESHOLD = 0.3
# Blocks whose largest span is at or below this size are footnotes / chart captions
# (corpus data: footnotes ≤ 8.4pt; body text ≥ 11pt; headings ≥ 11.4pt)
_FOOTNOTE_SIZE_THRESHOLD = 9.0


@dataclass
class PageContent:
    doc_name: str
    doc_title: str
    page_number: int        # 1-indexed
    text: str
    tables: list[list[list[str]]]
    table_titles: list[str]


def parse_document(path: Path) -> list[PageContent]:
    """Parse a PDF into per-page PageContent objects.

    Raises FileNotFoundError if the file is missing.
    Raises ValueError if the PDF is encrypted.
    """
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    doc_name = path.name

    with fitz.open(str(path)) as fitz_doc:
        if fitz_doc.is_encrypted:
            raise ValueError(f"PDF is encrypted: {path}")
        page_count = len(fitz_doc)
        doc_title = _detect_doc_title(fitz_doc, doc_name)

    pages: list[PageContent] = []
    with pdfplumber.open(str(path)) as pl_doc, fitz.open(str(path)) as fitz_doc:
        for idx in range(page_count):
            content = _process_page(
                pl_doc.pages[idx],
                fitz_doc[idx],
                doc_name,
                doc_title,
                page_num=idx + 1,
            )
            if content is not None:
                pages.append(content)

    logger.info("Parsed %d pages from %s", len(pages), doc_name)
    return pages


def _process_page(
    pl_page: pdfplumber.page.Page,
    fitz_page: fitz.Page,
    doc_name: str,
    doc_title: str,
    page_num: int,
) -> PageContent | None:
    tables: list[list[list[str]]] = []
    table_titles: list[str] = []
    table_bboxes: list[tuple[float, float, float, float]] = []

    try:
        tables, table_titles, table_bboxes = _extract_tables(pl_page, page_num, doc_name)
    except Exception as exc:
        logger.warning(
            "pdfplumber failed on page %d of %s, falling back to pymupdf: %s",
            page_num, doc_name, exc,
        )
        # table_bboxes stays empty → _extract_text runs without clipping

    text = _extract_text(fitz_page, table_bboxes)

    if not text.strip() and not tables:
        logger.warning("empty page %d in %s", page_num, doc_name)
        return None

    return PageContent(
        doc_name=doc_name,
        doc_title=doc_title,
        page_number=page_num,
        text=text,
        tables=tables,
        table_titles=table_titles,
    )


def _extract_tables(
    pl_page: pdfplumber.page.Page,
    page_num: int,
    doc_name: str,
) -> tuple[
    list[list[list[str]]],
    list[str],
    list[tuple[float, float, float, float]],
]:
    tables: list[list[list[str]]] = []
    titles: list[str] = []
    bboxes: list[tuple[float, float, float, float]] = []

    tbl_objects = pl_page.find_tables()
    if not tbl_objects:
        return tables, titles, bboxes

    raw_tables = pl_page.extract_tables()

    for tbl_obj, raw in zip(tbl_objects, raw_tables):
        if not raw:
            continue

        n_rows = len(raw)
        n_cols = len(raw[0]) if raw else 0

        if n_rows < 2:
            logger.warning(
                "Skipping table on page %d of %s: fewer than 2 rows", page_num, doc_name
            )
            continue
        if n_cols < 2:
            logger.warning(
                "Skipping table on page %d of %s: fewer than 2 columns", page_num, doc_name
            )
            continue

        total_cells = sum(len(row) for row in raw)
        empty_cells = sum(
            1 for row in raw for cell in row
            if cell is None or not str(cell).strip()
        )
        if total_cells and empty_cells / total_cells > 0.9:
            logger.warning(
                "Skipping table on page %d of %s: >90%% empty cells", page_num, doc_name
            )
            continue

        normalized: list[list[str]] = [
            [str(cell) if cell is not None else "" for cell in row]
            for row in raw
        ]
        tables.append(normalized)
        bboxes.append(tbl_obj.bbox)
        titles.append(_find_table_title(pl_page, tbl_obj.bbox))

    return tables, titles, bboxes


def _find_table_title(pl_page: pdfplumber.page.Page, table_bbox: tuple[float, float, float, float]) -> str:
    """Return the text line immediately above the table, or empty string."""
    x0, top_y, x1, _bottom_y = table_bbox
    search_top = max(0.0, top_y - _TITLE_SEARCH_GAP)
    search_bottom = top_y

    if search_top >= search_bottom:
        return ""

    try:
        region = pl_page.crop((x0, search_top, x1, search_bottom))
        text = (region.extract_text() or "").strip()
    except Exception as exc:
        logger.warning("Failed to extract table title for bbox %s: %s", table_bbox, exc)
        return ""

    if not text or len(text) > 150 or _is_header_footer_pattern(text):
        return ""
    return text


def _extract_text(
    fitz_page: fitz.Page,
    table_bboxes: list[tuple[float, float, float, float]],
) -> str:
    page_height = fitz_page.rect.height
    top_cutoff = page_height * _MARGIN_RATIO
    bottom_cutoff = page_height * (1.0 - _MARGIN_RATIO)

    parts: list[str] = []
    for block in fitz_page.get_text("dict")["blocks"]:
        if block.get("type") != 0:  # skip images and other non-text blocks
            continue
        x0, y0, x1, y1 = block["bbox"]
        mid_y = (y0 + y1) / 2.0
        if mid_y < top_cutoff or mid_y > bottom_cutoff:
            continue  # header or footer zone
        if table_bboxes and _overlaps_table((x0, y0, x1, y1), table_bboxes):
            continue
        max_size = max(
            (s["size"] for ln in block["lines"] for s in ln["spans"]),
            default=0.0,
        )
        if max_size <= _FOOTNOTE_SIZE_THRESHOLD:
            continue  # footnote or chart caption — font too small
        text = "\n".join(
            "".join(s["text"] for s in ln["spans"])
            for ln in block["lines"]
        )
        parts.append(text)

    return _clean_lines("\n".join(parts))


def _overlaps_table(
    bbox: tuple[float, float, float, float],
    table_bboxes: list[tuple[float, float, float, float]],
) -> bool:
    x0, y0, x1, y1 = bbox
    block_area = max(0.0, (x1 - x0) * (y1 - y0))
    if block_area == 0.0:
        return False
    for tx0, ty0, tx1, ty1 in table_bboxes:
        ix0, iy0 = max(x0, tx0), max(y0, ty0)
        ix1, iy1 = min(x1, tx1), min(y1, ty1)
        if ix0 < ix1 and iy0 < iy1:
            if (ix1 - ix0) * (iy1 - iy0) / block_area > _TABLE_OVERLAP_THRESHOLD:
                return True
    return False


def _clean_lines(text: str) -> str:
    """Strip page numbers and footnote/source lines from extracted text."""
    cleaned: list[str] = []
    for line in text.split("\n"):
        s = line.strip()
        if not s:
            continue
        if s.isdigit():
            continue  # page number
        if re.match(r"^(Note|Source|Sources)s?[:\.]", s, re.IGNORECASE):
            continue  # footnote or source label
        cleaned.append(s)
    return "\n".join(cleaned)


def _detect_doc_title(fitz_doc: fitz.Document, doc_name: str) -> str:
    """Return the largest-font text from the first two pages, or the filename stem."""
    candidates: list[tuple[float, str]] = []

    for page_idx in range(min(2, len(fitz_doc))):
        for block in fitz_doc[page_idx].get_text("dict")["blocks"]:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    size = float(span.get("size", 0))
                    if (
                        len(text) > 10
                        and not text.isdigit()
                        and not _is_header_footer_pattern(text)
                    ):
                        candidates.append((size, text))

    if not candidates:
        return Path(doc_name).stem

    candidates.sort(reverse=True)
    return candidates[0][1]


def _is_header_footer_pattern(text: str) -> bool:
    t = text.strip()
    if not t or len(t) < 5:
        return True
    if t.isdigit():
        return True
    # Date patterns common in FSR headers/footers: "12/2023", "12/03/2023"
    if re.match(r"^\d{1,2}[/\-\.](\d{1,2}[/\-\.])?\d{2,4}$", t):
        return True
    return False
