"""FastAPI application factory and process entry point."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI

from agentic_rag import __version__, db
from agentic_rag.answering import AnswerService
from agentic_rag.api import ask, documents, health
from agentic_rag.config import Settings, get_settings
from agentic_rag.embeddings import OllamaEmbedder
from agentic_rag.llm import OllamaProvider
from agentic_rag.models import EMBEDDING_DIM
from agentic_rag.retrieval import VectorRetriever


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)
    if settings.embedding_dim != EMBEDDING_DIM:
        raise RuntimeError(
            f"EMBEDDING_DIM={settings.embedding_dim} does not match the vector column "
            f"({EMBEDDING_DIM}). Changing it requires a migration and a full re-index."
        )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.engine = db.create_engine(str(settings.database_url))
        app.state.session_factory = db.create_session_factory(app.state.engine)
        app.state.http = httpx.AsyncClient(timeout=settings.http_timeout_s)
        embedder = OllamaEmbedder(
            app.state.http,
            settings.ollama_base_url,
            settings.embedding_model,
            settings.embedding_dim,
            settings.embed_batch_size,
        )
        llm = OllamaProvider(
            app.state.http, settings.ollama_base_url, settings.llm_model, settings.llm_num_ctx
        )
        app.state.answer_service = AnswerService(
            retriever=VectorRetriever(app.state.session_factory, embedder),
            llm=llm,
            top_k=settings.retrieval_top_k,
            temperature=settings.llm_temperature,
        )
        try:
            yield
        finally:
            await app.state.http.aclose()
            await app.state.engine.dispose()

    app = FastAPI(title="Agentic RAG", version=__version__, lifespan=lifespan)
    app.include_router(health.router)
    app.include_router(documents.router)
    app.include_router(ask.router)
    return app


app = create_app()
