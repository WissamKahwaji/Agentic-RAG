from collections.abc import AsyncIterator

import httpx
import pytest
from fastapi import FastAPI

from agentic_rag.config import get_settings
from agentic_rag.main import create_app


@pytest.fixture
def app() -> FastAPI:
    # Simple mode with dense retrieval keeps the test app from loading the 2 GB reranker.
    # Tracing off too: tests must not depend on (or send data to) a developer's Langfuse keys.
    overrides = {
        "retrieval_mode": "dense",
        "answer_mode": "simple",
        "langfuse_public_key": None,
        "langfuse_secret_key": None,
    }
    return create_app(get_settings().model_copy(update=overrides))


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    """In-process HTTP client. Runs the app lifespan so `app.state` resources exist."""
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c,
    ):
        yield c


@pytest.fixture(autouse=True)
def _no_tracing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test starts with tracing disabled, whatever the local .env contains."""
    from agentic_rag import tracing

    monkeypatch.setattr(tracing, "_client", None)
