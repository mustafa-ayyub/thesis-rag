#!/usr/bin/env python3
"""Ingest PDFs from data/documents/ into Qdrant.

Usage:
    python scripts/ingest.py
    python scripts/ingest.py --file data/documents/ecb_fsr_2023.pdf
    python scripts/ingest.py --export-chunks chunks.json
    python scripts/ingest.py --vectors-file vectors.json

Batch SLURM workflow:
    1. python scripts/ingest.py --export-chunks chunks.json   (laptop)
    2. scp chunks.json to cluster
    3. python cluster/batch_embed.py --input chunks.json --output vectors.json  (cluster)
    4. scp vectors.json back to laptop
    5. python scripts/ingest.py --vectors-file vectors.json   (laptop, re-parses for metadata)
"""
import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _collect_pdfs(cfg, single_file: str | None) -> list[Path]:
    if single_file:
        return [Path(single_file)]
    return sorted(Path(cfg.data.documents_dir).glob("*.pdf"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest PDFs into Qdrant.")
    parser.add_argument("--config", default="config/settings.yaml")
    parser.add_argument("--file", default=None, metavar="PDF")
    parser.add_argument(
        "--export-chunks", default=None, metavar="FILE",
        help="Parse+chunk only; write chunk JSON for batch embedding on the cluster.",
    )
    parser.add_argument(
        "--vectors-file", default=None, metavar="FILE",
        help="Load pre-computed vectors from cluster/batch_embed.py output.",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.settings import load_settings
    from src.ingestion.parser import parse_document
    from src.ingestion.chunker import chunk_document

    cfg = load_settings(args.config)
    pdfs = _collect_pdfs(cfg, args.file)

    if not pdfs:
        logger.error("No PDFs found.")
        sys.exit(1)

    logger.info("Found %d PDF(s)", len(pdfs))

    all_chunks = []
    for pdf in pdfs:
        logger.info("Parsing %s", pdf.name)
        try:
            pages = parse_document(pdf)
        except (FileNotFoundError, ValueError) as exc:
            logger.error("Skipping %s: %s", pdf.name, exc)
            continue
        chunks = chunk_document(pages, cfg)
        if not chunks:
            logger.warning("No chunks produced from %s", pdf.name)
            continue
        logger.info("  %s → %d pages, %d chunks", pdf.name, len(pages), len(chunks))
        all_chunks.extend(chunks)

    if not all_chunks:
        logger.error("No chunks produced from any PDF.")
        sys.exit(1)

    if args.export_chunks:
        out = Path(args.export_chunks)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            json.dump([asdict(c) for c in all_chunks], f)
        logger.info("Exported %d chunks to %s", len(all_chunks), out)
        return

    if args.vectors_file:
        from src.ingestion.indexer import index_chunks_precomputed

        with open(args.vectors_file) as f:
            raw: list[dict] = json.load(f)
        vectors = {
            item["chunk_id"]: (
                item["dense"],
                {int(k): float(v) for k, v in item["sparse"].items()},
            )
            for item in raw
        }
        index_chunks_precomputed(all_chunks, vectors, cfg)
    else:
        from src.ingestion.indexer import index_chunks
        from src.retrieval.embedder import Embedder

        embedder = Embedder(cfg)
        index_chunks(all_chunks, embedder, cfg)

    logger.info("Done. %d chunks indexed.", len(all_chunks))


if __name__ == "__main__":
    main()
