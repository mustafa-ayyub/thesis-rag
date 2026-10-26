import os
from pathlib import Path

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class GpuSettings(BaseModel):
    device: str = "cuda:0"
    batch_size_embed: int = 32
    batch_size_rerank: int = 16
    remote_url: str = ""  # "http://localhost:8000" when SSH tunnel is active


class ModelSettings(BaseModel):
    embedding: str = "BAAI/bge-m3"
    reranker: str = "BAAI/bge-reranker-v2-m3"


class SaiaSettings(BaseSettings):
    # api_key is read from the SAIA_API_KEY env var (env_prefix "SAIA_" +
    # field name), never from YAML
    model_config = SettingsConfigDict(env_prefix="SAIA_", env_file=".env", extra="ignore")

    api_key: str = ""
    base_url: str = "https://chat-ai.academiccloud.de/v1"
    model: str = "qwen3.8-27b"
    temperature: float = 0.0
    max_tokens: int = 2048
    rewriter_max_tokens: int = 300
    request_interval_seconds: float = 3.0
    # Safety margins below SAIA's official 30/min, 200/hour, 1000/day limits
    rate_limit_per_minute: int = 28
    rate_limit_per_hour: int = 180
    rate_limit_per_day: int = 900


class RetrievalSettings(BaseModel):
    chunk_size_tokens: int = 512
    chunk_overlap_tokens: int = 64
    hybrid_top_k: int = 20
    rerank_top_n: int = 5
    rrf_k: int = 60


class QdrantSettings(BaseModel):
    host: str = "localhost"
    port: int = 6333
    collection_name: str = "thesis_corpus"
    dense_vector_size: int = 1024
    dense_vector_name: str = "dense"
    sparse_vector_name: str = "sparse"


class DataSettings(BaseModel):
    documents_dir: str = "data/documents"
    queries_path: str = "data/queries/queries.json"
    results_dir: str = "data/results/retrieval"


class EvaluationSettings(BaseModel):
    system_a_results_dir: str = "data/results/system_a"
    answers_filename: str = "answers.jsonl"
    ragas_scores_filename: str = "ragas_scores.json"
    evaluation_final_filename: str = "evaluation_final.json"
    runs_per_query: int = 3
    generation_pace_seconds: float = 10.0
    generator_retry_attempts: int = 5
    generator_retry_wait_seconds: float = 30.0
    generator_min_answer_chars: int = 15
    ragas_pace_seconds: float = 3.0
    ragas_retry_attempts: int = 3
    ragas_retry_base_seconds: float = 10.0
    ragas_strictness: int = 3
    ragas_timeout_seconds: float = 300.0
    ragas_max_tokens: int = 4096


class SystemBSettings(BaseModel):
    results_dir: str = "data/results/system_b"
    answers_filename: str = "answers.jsonl"
    ragas_scores_filename: str = "ragas_scores.json"
    evaluation_final_filename: str = "evaluation_final.json"
    runs_per_query: int = 3
    max_retrieval_calls: int = 3
    max_final_chunks: int = 5
    agent_max_tokens: int = 800
    agent_retry_attempts: int = 3
    agent_retry_base_seconds: float = 10.0
    mcp_retry_attempts: int = 3
    mcp_retry_wait_seconds: float = 2.0
    generation_pace_seconds: float = 10.0
    generator_retry_attempts: int = 5
    generator_retry_wait_seconds: float = 30.0
    generator_min_answer_chars: int = 15


class Settings(BaseModel):
    gpu: GpuSettings = GpuSettings()
    models: ModelSettings = ModelSettings()
    saia: SaiaSettings = SaiaSettings()
    retrieval: RetrievalSettings = RetrievalSettings()
    qdrant: QdrantSettings = QdrantSettings()
    data: DataSettings = DataSettings()
    evaluation: EvaluationSettings = EvaluationSettings()
    system_b: SystemBSettings = SystemBSettings()


def load_settings(config_path: str = "config/settings.yaml") -> Settings:
    path = Path(config_path)
    raw: dict = {}
    if path.exists():
        with path.open() as f:
            raw = yaml.safe_load(f) or {}
    # api_key must never come from YAML — drop it so pydantic-settings fills it
    # from SAIA_API_KEY instead (YAML init values would otherwise win over env)
    saia_raw = raw.pop("saia", {})
    saia_raw.pop("api_key", None)
    # env overrides for per-machine deployment (Docker, different GPU hosts);
    # unset variables leave the YAML values untouched
    qdrant_raw = dict(raw.get("qdrant") or {})
    if host := os.environ.get("QDRANT_HOST"):
        qdrant_raw["host"] = host
    if port := os.environ.get("QDRANT_PORT"):
        qdrant_raw["port"] = int(port)
    raw["qdrant"] = qdrant_raw
    gpu_raw = dict(raw.get("gpu") or {})
    if remote_url := os.environ.get("GPU_REMOTE_URL"):
        gpu_raw["remote_url"] = remote_url
    raw["gpu"] = gpu_raw
    return Settings(saia=SaiaSettings(**saia_raw), **raw)
