"""Recursive text chunker and table-to-markdown converter.

Implements the algorithm from docs/chunking-strategy.md.
"""
import hashlib
import logging
import re

import tiktoken

from src.ingestion.parser import PageContent
from src.settings import Settings
from src.types import ChunkMetadata

logger = logging.getLogger(__name__)

# cl100k_base matches qwen8-27b's tokenizer, keeping chunk token counts
# consistent with generation without a second tokenization pass
_ENCODING = "cl100k_base"

MIN_CHUNK_TOKENS = 50

# (pattern_string, is_regex) — tried in priority order for splitting
_SEPARATORS: list[tuple[str, bool]] = [
    ("\n\n", False),                # paragraph break
    (r"(?<=\.) (?=[A-Z])", True),   # sentence end: ". Capital"
    (r"(?<=[;:]) ", True),          # clause boundary: "; " or ": "
    (" ", False),                   # word boundary
]

_HEADING_PATTERNS = [
    re.compile(r"^[A-Z][A-Z\s,\-:]{4,79}$"),                    # ALL CAPS, 5–80 chars
    re.compile(r"^\d+(\.\d+)?\s+[A-Z]"),                         # numbered section
    re.compile(r"^(Chapter|Section|Part)\s+\d+", re.IGNORECASE), # chapter label
]


def chunk_document(pages: list[PageContent], cfg: Settings) -> list[ChunkMetadata]:
    """Chunk parsed pages into ChunkMetadata objects.

    Tracks section_title and chapter state across pages within the document.
    """
    enc = tiktoken.get_encoding(_ENCODING)
    max_tokens = cfg.retrieval.chunk_size_tokens
    overlap_tokens = cfg.retrieval.chunk_overlap_tokens

    chunks: list[ChunkMetadata] = []
    section_title = ""
    chapter = ""

    for page in pages:
        section_title, chapter = _update_heading_state(
            page.text, section_title, chapter
        )

        for text in _chunk_text(page.text, max_tokens, overlap_tokens, enc):
            chunks.append(ChunkMetadata(
                chunk_id=_make_chunk_id(page.doc_name, page.page_number, text),
                doc_name=page.doc_name,
                doc_title=page.doc_title,
                page_number=page.page_number,
                chunk_type="text",
                section_title=section_title,
                chapter=chapter,
                token_count=_count_tokens(text, enc),
                text=text,
            ))

        for table, title in zip(page.tables, page.table_titles):
            markdown = _table_to_markdown(table, page.page_number, page.doc_name)
            if markdown is None:
                continue
            effective_title = title or (
                f"{section_title} (table)" if section_title else "(table)"
            )
            full_text = f"{effective_title}\n{markdown}"
            chunks.append(ChunkMetadata(
                chunk_id=_make_chunk_id(page.doc_name, page.page_number, full_text),
                doc_name=page.doc_name,
                doc_title=page.doc_title,
                page_number=page.page_number,
                chunk_type="table",
                section_title=section_title,
                chapter=chapter,
                token_count=_count_tokens(full_text, enc),
                text=full_text,
            ))

    logger.info("Produced %d chunks from %d pages", len(chunks), len(pages))
    return chunks


def _chunk_text(
    text: str,
    max_tokens: int,
    overlap_tokens: int,
    enc: tiktoken.Encoding,
) -> list[str]:
    atoms = _split_atoms(text, max_tokens, enc)
    return _assemble(atoms, max_tokens, overlap_tokens, enc)


def _split_atoms(
    text: str,
    max_tokens: int,
    enc: tiktoken.Encoding,
    sep_index: int = 0,
) -> list[str]:
    """Recursively split text into atoms ≤ max_tokens using separator hierarchy."""
    if not text.strip():
        return []
    if _count_tokens(text, enc) <= max_tokens:
        return [text]

    if sep_index >= len(_SEPARATORS):
        return _word_split(text, max_tokens, enc)

    sep, is_regex = _SEPARATORS[sep_index]
    pieces = re.split(sep, text) if is_regex else text.split(sep)
    pieces = [p.strip() for p in pieces if p.strip()]

    if len(pieces) <= 1:
        return _split_atoms(text, max_tokens, enc, sep_index + 1)

    atoms: list[str] = []
    for piece in pieces:
        if _count_tokens(piece, enc) > max_tokens:
            atoms.extend(_split_atoms(piece, max_tokens, enc, sep_index + 1))
        else:
            atoms.append(piece)
    return atoms


def _word_split(
    text: str, max_tokens: int, enc: tiktoken.Encoding
) -> list[str]:
    """Hard split at word boundaries when all other separators are exhausted."""
    words = text.split(" ")
    atoms: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for word in words:
        w_tokens = _count_tokens(word, enc)
        if current and current_tokens + w_tokens > max_tokens:
            atoms.append(" ".join(current))
            current = [word]
            current_tokens = w_tokens
        else:
            current.append(word)
            current_tokens += w_tokens
    if current:
        atoms.append(" ".join(current))
    return atoms


def _assemble(
    atoms: list[str],
    max_tokens: int,
    overlap_tokens: int,
    enc: tiktoken.Encoding,
) -> list[str]:
    """Pack atoms into chunks of ≤ max_tokens with overlap between chunks."""
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for atom in atoms:
        atom_tokens = _count_tokens(atom, enc)
        sep_tokens = 1 if current else 0  # "\n" joining atom to existing atoms

        if current and current_tokens + sep_tokens + atom_tokens > max_tokens:
            text = "\n".join(current)
            if _count_tokens(text, enc) >= MIN_CHUNK_TOKENS:
                chunks.append(text)

            # Back-fill overlap from the tail of the flushed chunk
            overlap: list[str] = []
            overlap_count = 0
            for p in reversed(current):
                p_tokens = _count_tokens(p, enc)
                if overlap_count + p_tokens <= overlap_tokens:
                    overlap.insert(0, p)
                    overlap_count += p_tokens
                else:
                    break
            current = overlap + [atom]
            # separators between the len(current) atoms = len(current) - 1
            current_tokens = overlap_count + atom_tokens + len(overlap)
        else:
            current.append(atom)
            current_tokens += sep_tokens + atom_tokens

    if current:
        text = "\n".join(current)
        if _count_tokens(text, enc) >= MIN_CHUNK_TOKENS:
            chunks.append(text)

    return chunks


def _table_to_markdown(
    table: list[list[str]],
    page_num: int,
    doc_name: str,
) -> str | None:
    """Convert a table to markdown.  Returns None on failure (already logged)."""
    try:
        header = table[0]
        rows = table[1:]
        n_cols = len(header)

        lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join("---" for _ in header) + " |",
        ]
        for row in rows:
            # Pad or truncate row to match header column count
            padded = (list(row) + [""] * n_cols)[:n_cols]
            lines.append("| " + " | ".join(padded) + " |")

        return "\n".join(lines)
    except Exception as exc:
        logger.warning(
            "Table markdown conversion failed on page %d of %s: %s",
            page_num, doc_name, exc,
        )
        return None


def _update_heading_state(
    text: str,
    current_title: str,
    current_chapter: str,
) -> tuple[str, str]:
    """Scan text lines for a heading and return updated (section_title, chapter).

    Takes the last heading on the page so cross-page state carries the section
    a page ends in, not the one it starts in.
    """
    title, chapter = current_title, current_chapter
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        for pattern in _HEADING_PATTERNS:
            if pattern.match(line):
                title, chapter = line, _extract_chapter(line)
                break
    return title, chapter


def _extract_chapter(section_title: str) -> str:
    """Return chapter label from a numbered heading, e.g. '2 Financial markets'."""
    m = re.match(r"^(\d+(?:\.\d+)?)\s+(.*)", section_title)
    if not m:
        return ""
    prefix = m.group(1)
    rest_words = m.group(2).split()[:6]  # numeric prefix + up to 6 following words
    return f"{prefix} {' '.join(rest_words)}".strip()


def _count_tokens(text: str, enc: tiktoken.Encoding) -> int:
    return len(enc.encode(text))


def _make_chunk_id(doc_name: str, page_number: int, text: str) -> str:
    key = f"{doc_name}|{page_number}|{text}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]
