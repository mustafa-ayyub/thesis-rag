#!/usr/bin/env python3
"""Inspect how parser+chunker render specific pages, table chunks in full.

Runs the exact ingestion code path (_process_page -> chunk_document) on
individual pages only — no full-document parse, no Qdrant, no GPU.

Usage:
    python scripts/inspect_table_pages.py
    python scripts/inspect_table_pages.py --doc ECB-Financial-Stability-Reviews-2026.pdf --page 90
    python scripts/inspect_table_pages.py --doc Bundesbank-Financial-Stability-Review-2023.pdf --page 24
"""
import argparse
import sys
from pathlib import Path

import fitz
import pdfplumber

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.settings import load_settings
from src.ingestion.parser import _detect_doc_title, _process_page
from src.ingestion.chunker import chunk_document

DOCS_DIR = Path("data/documents")

DEFAULT_PAGES = [
    ("Bundesbank-Financial-Stability-Review-2023.pdf", 94),
    ("ECB-Financial-Stability-Reviews-2026.pdf", 90),
    ("ECB-Financial-Stability-Reviews-2026.pdf", 124),
    ("ECB-Financial-Stability-Reviews-2026.pdf", 129),
    ("ECB-Financial-Stability-Reviews-2026.pdf", 130),
]


def inspect_page(doc_name: str, page_no: int, cfg) -> None:
    path = DOCS_DIR / doc_name
    with fitz.open(str(path)) as fitz_doc:
        doc_title = _detect_doc_title(fitz_doc, doc_name)
    with pdfplumber.open(str(path)) as pl_doc, fitz.open(str(path)) as fitz_doc:
        content = _process_page(
            pl_doc.pages[page_no - 1], fitz_doc[page_no - 1],
            doc_name, doc_title, page_num=page_no,
        )

    print(f"================ {doc_name}  page {page_no} ================")
    if content is None:
        print("  (empty page)")
        return
    print(f"  raw text: {len(content.text.split())} words | "
          f"tables detected: {len(content.tables)} | titles: {content.table_titles!r}")

    chunks = chunk_document([content], cfg)
    for c in chunks:
        head = (f"  --- {c.chunk_type} chunk | {c.token_count} tokens | "
                f"section={c.section_title!r}")
        if c.chunk_type == "table":
            print(head + "\n" + c.text)
        else:
            print(head + f" | text starts: {c.text[:100]!r}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc", default=None, help="PDF filename in data/documents/")
    parser.add_argument("--page", type=int, default=None, help="1-indexed physical page")
    args = parser.parse_args()

    if args.doc and args.page:
        targets = [(args.doc, args.page)]
    elif args.doc or args.page:
        parser.error("--doc and --page must be given together")
    else:
        targets = DEFAULT_PAGES

    cfg = load_settings("config/settings.yaml")
    for doc_name, page_no in targets:
        inspect_page(doc_name, page_no, cfg)


if __name__ == "__main__":
    main()
