"""Application settings, loaded from environment variables and the repo-root `.env`."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, PostgresDsn, SecretStr
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

    # Where the LLM runs: "ollama" (local) or any OpenAI-compatible API (Groq, Gemini, ...).
    llm_provider: Literal["ollama", "openai_compatible"] = "ollama"
    llm_base_url: str = "https://api.groq.com/openai/v1"
    # Accepted as LLM_API_KEY or GROQ_API_KEY. Never logged.
    llm_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("LLM_API_KEY", "GROQ_API_KEY")
    )
    llm_max_tokens: int = Field(default=1536, ge=64)
    # Retries on HTTP 429. Free tiers (Groq: 8k tokens/minute) need more patience in batch runs.
    llm_max_retries: int = Field(default=6, ge=0, le=20)
    # Where embeddings run: "ollama", or "local" (same model in-process; no Ollama needed).
    embedding_provider: Literal["ollama", "local"] = "ollama"
    # Hugging Face id of the model EMBEDDING_MODEL names in Ollama (bge-m3 in both).
    local_embedding_model: str = "BAAI/bge-m3"
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
    # None = Apple GPU (MPS) when available, else CPU. Set "cpu" to measure CPU-only hosting.
    reranker_device: str | None = None

    # "simple": one retrieve-then-generate pass. "agentic": the LangGraph agent (agent/graph.py).
    answer_mode: Literal["simple", "agentic"] = "agentic"
    # Reranker score a passage needs to reach the answer step (calibrated: make eval-grading).
    relevance_threshold: float = Field(default=0.05, ge=0.0, le=1.0)
    max_retrieval_attempts: int = Field(default=2, ge=1, le=5)
    max_generation_attempts: int = Field(default=2, ge=1, le=5)
    self_check: bool = True

    # Default knowledge for a question: "documents" only, or "hybrid" (router + web fallback +
    # model knowledge). Requests can override it. Hybrid requires answer_mode="agentic".
    knowledge_mode: Literal["documents", "hybrid"] = "documents"
    # One sentence the router uses to judge whether a question is in scope of the documents.
    corpus_description: str = (
        "the pgvector Postgres extension, the FastAPI web framework (tutorial and deployment "
        "docs), and the 2020 paper on Retrieval-Augmented Generation (RAG)"
    )
    # Keyless metasearch backends tried in order; Tavily is used instead when a key is set.
    web_search_backends: str = "auto,yahoo,brave,duckduckgo"
    web_results: int = Field(default=5, ge=1, le=10)
    tavily_api_key: str | None = None

    http_timeout_s: float = 60.0

    # Tracing (optional): set both keys to send traces to Langfuse (self-hosted or cloud).
    langfuse_public_key: str | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_host: str = "http://localhost:3100"


@lru_cache
def get_settings() -> Settings:
    """Process-wide settings singleton. Override in tests via `app.dependency_overrides`."""
    return Settings()
