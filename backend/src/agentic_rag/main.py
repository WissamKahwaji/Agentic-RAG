"""FastAPI application factory and process entry point."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from agentic_rag import __version__, db, tracing
from agentic_rag.agent import AgentAnswerService, AgentConfig
from agentic_rag.answering import AnswerService, QuestionAnswerer
from agentic_rag.api import ask, documents, health
from agentic_rag.config import Settings, get_settings
from agentic_rag.embeddings import Embedder
from agentic_rag.llm import LLMProvider
from agentic_rag.models import EMBEDDING_DIM
from agentic_rag.providers import build_embedder, build_llm, build_reranker
from agentic_rag.reranking import Reranker
from agentic_rag.search import build_retriever
from agentic_rag.web import MetaWebSearch, TavilyWebSearch, WebSearch


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )


async def build_answer_service(
    settings: Settings,
    session_factory: async_sessionmaker[AsyncSession],
    embedder: Embedder,
    llm: LLMProvider,
    http: httpx.AsyncClient,
    reranker: Reranker | None = None,
) -> QuestionAnswerer:
    needs_reranker = settings.answer_mode == "agentic" or "rerank" in settings.retrieval_mode
    if needs_reranker and reranker is None:
        model = build_reranker(settings)
        await model.warm_up()  # load the model now, so the first question doesn't pay for it
        reranker = model

    if settings.answer_mode == "agentic":
        assert reranker is not None
        web: WebSearch = (
            TavilyWebSearch(http, settings.tavily_api_key)
            if settings.tavily_api_key
            else MetaWebSearch([b.strip() for b in settings.web_search_backends.split(",")])
        )
        return AgentAnswerService(
            # The agent grades passages itself, so it takes the first-stage retriever only.
            retriever=build_retriever("hybrid", session_factory, embedder, settings),
            reranker=reranker,
            llm=llm,
            top_k=settings.retrieval_top_k,
            config=AgentConfig(
                candidates=settings.rerank_candidates,
                relevance_threshold=settings.relevance_threshold,
                max_retrieval_attempts=settings.max_retrieval_attempts,
                max_generation_attempts=settings.max_generation_attempts,
                self_check=settings.self_check,
                temperature=settings.llm_temperature,
                web_results=settings.web_results,
                corpus_description=settings.corpus_description,
            ),
            web=web,
        )
    return AnswerService(
        retriever=build_retriever(
            settings.retrieval_mode, session_factory, embedder, settings, reranker
        ),
        llm=llm,
        top_k=settings.retrieval_top_k,
        temperature=settings.llm_temperature,
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
        tracing.init_tracing(settings)
        app.state.engine = db.create_engine(str(settings.database_url))
        app.state.session_factory = db.create_session_factory(app.state.engine)
        app.state.http = httpx.AsyncClient(timeout=settings.http_timeout_s)
        embedder = build_embedder(settings, app.state.http)
        llm = build_llm(settings, app.state.http)
        app.state.answer_service = await build_answer_service(
            settings, app.state.session_factory, embedder, llm, app.state.http
        )
        try:
            yield
        finally:
            tracing.flush()
            await app.state.http.aclose()
            await app.state.engine.dispose()

    app = FastAPI(title="Agentic RAG", version=__version__, lifespan=lifespan)
    app.include_router(health.router)
    app.include_router(documents.router)
    app.include_router(ask.router)
    return app


app = create_app()
