"""Build the model-backed components from settings, so every entry point (API, ingestion CLI,
evals) gets the same LLM, embedder and reranker for a given configuration.

Local development:  LLM_PROVIDER=ollama,            EMBEDDING_PROVIDER=ollama
CPU-only hosting:   LLM_PROVIDER=openai_compatible,  EMBEDDING_PROVIDER=local,
                    RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2, RELEVANCE_THRESHOLD=0.10
"""

import httpx

from agentic_rag.config import Settings
from agentic_rag.embeddings import Embedder, OllamaEmbedder, SentenceTransformerEmbedder
from agentic_rag.llm import LLMProvider, OllamaProvider, OpenAICompatibleProvider
from agentic_rag.reranking import CrossEncoderReranker


class ConfigurationError(RuntimeError):
    pass


def build_llm(settings: Settings, http: httpx.AsyncClient) -> LLMProvider:
    if settings.llm_provider == "openai_compatible":
        if settings.llm_api_key is None:
            raise ConfigurationError(
                "LLM_PROVIDER=openai_compatible needs LLM_API_KEY (or GROQ_API_KEY) in .env"
            )
        return OpenAICompatibleProvider(
            http,
            settings.llm_base_url,
            settings.llm_api_key.get_secret_value(),
            settings.llm_model,
            max_tokens=settings.llm_max_tokens,
        )
    return OllamaProvider(
        http,
        settings.ollama_base_url,
        settings.llm_model,
        settings.llm_num_ctx,
        max_tokens=settings.llm_max_tokens,
    )


def build_embedder(settings: Settings, http: httpx.AsyncClient) -> Embedder:
    if settings.embedding_provider == "local":
        return SentenceTransformerEmbedder(
            settings.local_embedding_model, settings.embedding_dim, name=settings.embedding_model
        )
    return OllamaEmbedder(
        http,
        settings.ollama_base_url,
        settings.embedding_model,
        settings.embedding_dim,
        settings.embed_batch_size,
    )


def build_reranker(settings: Settings) -> CrossEncoderReranker:
    return CrossEncoderReranker(settings.reranker_model, device=settings.reranker_device)


def uses_ollama(settings: Settings) -> bool:
    return settings.llm_provider == "ollama" or settings.embedding_provider == "ollama"
