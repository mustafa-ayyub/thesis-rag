#!/usr/bin/env python3
"""Export each PDF in data/documents/ to a plain-text file with page markers.

Page markers use the PHYSICAL page number (1-indexed, as seen in a PDF
viewer) — the same convention used by queries.json, chunks, and validation.
The printed page number (usually in the page header/footer) remains in the
text where the PDF places it.

Usage:
    python scripts/export_pages_txt.py
    python scripts/export_pages_txt.py --out data/pages_txt
"""
import argparse
from pathlib import Path

import pymupdf

_SEPARATOR = "=" * 78


def export_pdf(pdf_path: Path, out_path: Path) -> int:
    doc = pymupdf.open(pdf_path)
    lines = [
        f"FILE: {pdf_path.name}  ({len(doc)} physical pages)",
        "Page markers below are PHYSICAL page numbers (1-indexed).",
        "",
    ]
    for i in range(len(doc)):
        text = doc[i].get_text("text").rstrip()
        lines.append(_SEPARATOR)
        lines.append(f"PAGE {i + 1}")
        lines.append(_SEPARATOR)
        lines.append(text if text else "(no extractable text)")
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return len(doc)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents-dir", default="data/documents")
    parser.add_argument("--out", default="data/pages_txt")
    args = parser.parse_args()

    docs_dir = Path(args.documents_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for pdf in sorted(docs_dir.glob("*.pdf")):
        out_path = out_dir / (pdf.stem + ".txt")
        n = export_pdf(pdf, out_path)
        print(f"{pdf.name} -> {out_path}  ({n} pages, {out_path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
