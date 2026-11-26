from collections.abc import AsyncIterator

import httpx
import pytest
from fastapi import FastAPI

from agentic_rag.config import get_settings
from agentic_rag.main import create_app


@pytest.fixture
def app() -> FastAPI:
    # Dense retrieval keeps the test app from loading the 2 GB reranker model.
    return create_app(get_settings().model_copy(update={"retrieval_mode": "dense"}))


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    """In-process HTTP client. Runs the app lifespan so `app.state` resources exist."""
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c,
    ):
        yield c
