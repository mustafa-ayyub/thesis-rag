#!/usr/bin/env python3
"""
Batch embedding job for SLURM.

Reads text chunks from a JSON file, embeds them with BGE-M3,
and writes the dense + sparse vectors to an output JSON file.

Usage:
    python cluster/batch_embed.py --input chunks.json --output vectors.json

Input format  (list of objects, must have "chunk_id" and "text"):
    [{"chunk_id": "abc123", "text": "...", ...}, ...]

Output format:
    [{"chunk_id": "abc123", "dense": [...], "sparse": {"42": 0.8, ...}}, ...]

The indexer on the laptop then reads vectors.json and upserts into Qdrant.
"""
import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="BAAI/bge-m3")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    from FlagEmbedding import BGEM3FlagModel
    import torch

    logger.info("Loading %s", args.model)
    model = BGEM3FlagModel(args.model, use_fp16=True)

    with open(args.input) as f:
        items: list[dict] = json.load(f)

    if not items:
        logger.warning("Input file is empty")
        Path(args.output).write_text("[]")
        sys.exit(0)

    texts = [item["text"] for item in items]
    logger.info("Embedding %d texts with batch_size=%d", len(texts), args.batch_size)

    results: list[dict] = []
    batch_size = args.batch_size

    i = 0
    while i < len(texts):
        batch = texts[i : i + batch_size]
        try:
            output = model.encode(
                batch,
                batch_size=batch_size,
                max_length=512,
                return_dense=True,
                return_sparse=True,
            )
        except torch.cuda.OutOfMemoryError:
            if batch_size == 1:
                raise
            batch_size = max(1, batch_size // 2)
            logger.warning("OOM — reducing batch_size to %d and retrying", batch_size)
            torch.cuda.empty_cache()
            continue  # retry same position with the smaller (persisted) batch

        for j, (dense, sparse) in enumerate(
            zip(output["dense_vecs"], output["lexical_weights"])
        ):
            results.append({
                "chunk_id": items[i + j]["chunk_id"],
                "dense": dense.tolist(),
                "sparse": {str(k): float(v) for k, v in sparse.items()},
            })

        i += batch_size
        logger.info("Progress: %d / %d", min(i, len(texts)), len(texts))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f)
    logger.info("Wrote %d vectors → %s", len(results), args.output)


if __name__ == "__main__":
    main()
