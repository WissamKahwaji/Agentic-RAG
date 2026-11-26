"""FastAPI application factory and process entry point."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI

from agentic_rag import __version__, db
from agentic_rag.api import health
from agentic_rag.config import Settings, get_settings


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.engine = db.create_engine(str(settings.database_url))
        app.state.http = httpx.AsyncClient(timeout=settings.http_timeout_s)
        try:
            yield
        finally:
            await app.state.http.aclose()
            await app.state.engine.dispose()

    app = FastAPI(title="Agentic RAG", version=__version__, lifespan=lifespan)
    app.include_router(health.router)
    return app


app = create_app()
