"""Application settings, loaded from environment variables and the repo-root `.env`."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, PostgresDsn
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[3]

# Named after their stages: first-stage retrieval, then optional cross-encoder reranking.
RetrievalMode = Literal["dense", "keyword", "hybrid", "dense+rerank", "hybrid+rerank"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: str = Field(default="local", description="local | test | production")
    log_level: str = "INFO"

    database_url: PostgresDsn = PostgresDsn("postgresql+psycopg://rag:rag@localhost:5432/rag")

    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "qwen2.5:7b"
    # 5 chunks x ~300 tokens + instructions + answer fits comfortably in 8k.
    llm_num_ctx: int = Field(default=8192, ge=2048)
    llm_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    embedding_model: str = "bge-m3"
    embedding_dim: int = 1024
    embed_batch_size: int = Field(default=32, ge=1)

    # ~1200 chars is roughly 300 tokens: small enough for precise retrieval, large enough to
    # keep a complete thought. Overlap stops sentences being cut in half at chunk boundaries.
    chunk_size: int = Field(default=1200, ge=200)
    chunk_overlap: int = Field(default=200, ge=0)

    retrieval_top_k: int = Field(default=5, ge=1, le=20)
    # Chosen by the retrieval eval (`make eval-retrieval`): best hit@1/MRR, ~350 ms extra.
    retrieval_mode: RetrievalMode = "hybrid+rerank"
    # How many results each retriever contributes before fusion.
    hybrid_candidates: int = Field(default=30, ge=5, le=200)
    # Cross-encoder that re-orders the first-stage candidates (only used by "+rerank" modes).
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    # 10 matched 20 and 30 on hit@1/MRR at half the latency; raise it for large corpora.
    rerank_candidates: int = Field(default=10, ge=5, le=100)

    http_timeout_s: float = 60.0


@lru_cache
def get_settings() -> Settings:
    """Process-wide settings singleton. Override in tests via `app.dependency_overrides`."""
    return Settings()
