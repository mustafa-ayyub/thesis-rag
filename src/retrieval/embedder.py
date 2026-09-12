import logging

import httpx

from src.settings import Settings
from src.types import ScoredChunk, RetrievedChunk

logger = logging.getLogger(__name__)

_BGE_MAX_LENGTH = 512

SparseVector = dict[int, float]
DenseVector = list[float]


class Embedder:
    """
    Wraps BGE-M3 for dense + sparse embedding.

    Mode is determined by cfg.gpu.remote_url:
    - Empty string  → load model locally (runs on the GPU cluster)
    - Non-empty URL → forward requests to the FastAPI server via HTTP
                      (laptop mode with the SSH tunnel active)
    """

    def __init__(self, cfg: Settings) -> None:
        self._cfg = cfg
        self._remote_url: str = cfg.gpu.remote_url.rstrip("/")
        self._model = None
        if not self._remote_url:
            self._load_model()

    def _load_model(self) -> None:
        from FlagEmbedding import BGEM3FlagModel

        logger.info("Loading %s on %s", self._cfg.models.embedding, self._cfg.gpu.device)
        self._model = BGEM3FlagModel(
            self._cfg.models.embedding,
            use_fp16=True,
            device=self._cfg.gpu.device,
        )

    def embed_query(self, query: str) -> tuple[DenseVector, SparseVector]:
        if self._remote_url:
            return self._embed_remote(query)
        return self._embed_local([query])[0]

    def embed_batch(self, texts: list[str]) -> list[tuple[DenseVector, SparseVector]]:
        """Embed multiple texts (dense + sparse)."""
        if self._remote_url:
            return self._embed_batch_remote(texts)
        return self._embed_local(texts)

    def _embed_local(self, texts: list[str]) -> list[tuple[DenseVector, SparseVector]]:
        import torch

        batch_size = self._cfg.gpu.batch_size_embed
        results: list[tuple[DenseVector, SparseVector]] = []

        i = 0
        while i < len(texts):
            batch = texts[i : i + batch_size]
            try:
                output = self._model.encode(
                    batch,
                    batch_size=batch_size,
                    max_length=_BGE_MAX_LENGTH,
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

            for dense, sparse_weights in zip(
                output["dense_vecs"], output["lexical_weights"]
            ):
                results.append((
                    dense.tolist(),
                    {int(k): float(v) for k, v in sparse_weights.items()},
                ))
            i += batch_size
            logger.debug("Embedded %d/%d texts", min(i, len(texts)), len(texts))

        return results

    def _embed_remote(self, query: str) -> tuple[DenseVector, SparseVector]:
        resp = httpx.post(
            f"{self._remote_url}/embed",
            json={"text": query},
            timeout=30.0,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["dense"], {int(k): float(v) for k, v in data["sparse"].items()}

    def _embed_batch_remote(
        self, texts: list[str]
    ) -> list[tuple[DenseVector, SparseVector]]:
        resp = httpx.post(
            f"{self._remote_url}/embed_batch",
            json={"texts": texts},
            timeout=300.0,
        )
        resp.raise_for_status()
        return [
            (item["dense"], {int(k): float(v) for k, v in item["sparse"].items()})
            for item in resp.json()["results"]
        ]
