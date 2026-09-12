import logging
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from FlagEmbedding import BGEM3FlagModel, FlagReranker
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

_embed_model = None
_rerank_model = None

_EMBED_BATCH_SIZE = int(os.environ.get("EMBED_BATCH_SIZE", "32"))
_RERANK_BATCH_SIZE = int(os.environ.get("RERANK_BATCH_SIZE", "16"))
_BGE_MAX_LENGTH = 512


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _embed_model, _rerank_model

    logger.info("Loading BAAI/bge-m3...")
    _embed_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
    logger.info("Loading BAAI/bge-reranker-v2-m3...")
    _rerank_model = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)
    logger.info("Both models ready.")
    yield
    _embed_model = None
    _rerank_model = None


app = FastAPI(title="Thesis GPU Server", lifespan=lifespan)


class EmbedRequest(BaseModel):
    text: str


class EmbedBatchRequest(BaseModel):
    texts: list[str]


class RerankCandidate(BaseModel):
    text: str
    index: int


class RerankRequest(BaseModel):
    query: str
    candidates: list[RerankCandidate]
    top_n: int


def _encode(texts: list[str]) -> list[dict[str, Any]]:
    import torch

    try:
        output = _embed_model.encode(
            texts,
            batch_size=_EMBED_BATCH_SIZE,
            max_length=_BGE_MAX_LENGTH,
            return_dense=True,
            return_sparse=True,
        )
    except torch.cuda.OutOfMemoryError:
        logger.warning("OOM at batch_size=%d, retrying at %d", _EMBED_BATCH_SIZE, _EMBED_BATCH_SIZE // 2)
        torch.cuda.empty_cache()
        output = _embed_model.encode(
            texts,
            batch_size=_EMBED_BATCH_SIZE // 2,
            max_length=_BGE_MAX_LENGTH,
            return_dense=True,
            return_sparse=True,
        )
    return [
        {
            "dense": dense.tolist(),
            "sparse": {str(k): float(v) for k, v in sparse.items()},
        }
        for dense, sparse in zip(output["dense_vecs"], output["lexical_weights"])
    ]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/embed")
def embed(req: EmbedRequest) -> dict[str, Any]:
    if _embed_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    result = _encode([req.text])[0]
    return {"dense": result["dense"], "sparse": result["sparse"]}


@app.post("/embed_batch")
def embed_batch(req: EmbedBatchRequest) -> dict[str, Any]:
    if _embed_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not req.texts:
        return {"results": []}
    return {"results": _encode(req.texts)}


@app.post("/rerank")
def rerank(req: RerankRequest) -> dict[str, Any]:
    if _rerank_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    import torch

    pairs = [(req.query, c.text) for c in req.candidates]
    try:
        scores: list[float] = _rerank_model.compute_score(pairs, batch_size=_RERANK_BATCH_SIZE)
    except torch.cuda.OutOfMemoryError:
        logger.warning("OOM at batch_size=%d, retrying at %d", _RERANK_BATCH_SIZE, _RERANK_BATCH_SIZE // 2)
        torch.cuda.empty_cache()
        scores = _rerank_model.compute_score(pairs, batch_size=_RERANK_BATCH_SIZE // 2)

    ranked = sorted(zip(req.candidates, scores), key=lambda x: x[1], reverse=True)
    return {
        "results": [
            {"index": c.index, "score": float(s)} for c, s in ranked[: req.top_n]
        ]
    }
