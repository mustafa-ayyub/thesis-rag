"""Embed chunks and upsert into Qdrant.

Creates the collection with dense + sparse vector config if it does not exist.
"""
import logging
from dataclasses import asdict

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from src.retrieval.embedder import Embedder
from src.settings import Settings
from src.types import ChunkMetadata

logger = logging.getLogger(__name__)

_LOG_EVERY = 100


def index_chunks(chunks: list[ChunkMetadata], embedder: Embedder, cfg: Settings) -> None:
    """Embed all chunks and upsert into Qdrant."""
    if not chunks:
        return
    client = QdrantClient(host=cfg.qdrant.host, port=cfg.qdrant.port)
    _ensure_collection(client, cfg)
    batch_size = cfg.gpu.batch_size_embed
    total = len(chunks)
    upserted = 0
    for i in range(0, total, batch_size):
        batch = chunks[i : i + batch_size]
        embeddings = embedder.embed_batch([c.text for c in batch])
        _upsert_batch(client, cfg, batch, embeddings)
        upserted += len(batch)
        if upserted % _LOG_EVERY == 0 or upserted == total:
            logger.info("Indexed %d/%d chunks", upserted, total)


def index_chunks_precomputed(
    chunks: list[ChunkMetadata],
    vectors: dict[str, tuple[list[float], dict[int, float]]],
    cfg: Settings,
) -> None:
    """Upsert chunks using pre-computed vectors (from cluster/batch_embed.py output)."""
    if not chunks:
        return
    matched = [c for c in chunks if c.chunk_id in vectors]
    if len(matched) < len(chunks):
        logger.warning(
            "%d chunks missing from vectors file — skipping them",
            len(chunks) - len(matched),
        )
    if not matched:
        return
    client = QdrantClient(host=cfg.qdrant.host, port=cfg.qdrant.port)
    _ensure_collection(client, cfg)
    embeddings = [vectors[c.chunk_id] for c in matched]
    _upsert_batch(client, cfg, matched, embeddings)
    logger.info("Indexed %d chunks from precomputed vectors", len(matched))


def _upsert_batch(
    client: QdrantClient,
    cfg: Settings,
    chunks: list[ChunkMetadata],
    embeddings: list[tuple[list[float], dict[int, float]]],
) -> None:
    points = [
        PointStruct(
            id=int(chunk.chunk_id, 16),
            vector={
                cfg.qdrant.dense_vector_name: dense,
                cfg.qdrant.sparse_vector_name: SparseVector(
                    indices=list(sparse_dict.keys()),
                    values=list(sparse_dict.values()),
                ),
            },
            payload=asdict(chunk),
        )
        for chunk, (dense, sparse_dict) in zip(chunks, embeddings)
    ]
    client.upsert(collection_name=cfg.qdrant.collection_name, points=points)


def _ensure_collection(client: QdrantClient, cfg: Settings) -> None:
    if client.collection_exists(cfg.qdrant.collection_name):
        return
    client.create_collection(
        collection_name=cfg.qdrant.collection_name,
        vectors_config={
            cfg.qdrant.dense_vector_name: VectorParams(
                size=cfg.qdrant.dense_vector_size,
                distance=Distance.COSINE,
            ),
        },
        sparse_vectors_config={
            cfg.qdrant.sparse_vector_name: SparseVectorParams(
                index=SparseIndexParams(on_disk=False),
            ),
        },
    )
    logger.info("Created Qdrant collection %r", cfg.qdrant.collection_name)

