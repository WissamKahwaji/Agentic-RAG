"""Liveness/readiness endpoint: reports the status of each downstream dependency."""

import logging
from collections.abc import Awaitable, Callable
from typing import Annotated, Literal

import httpx
from fastapi import APIRouter, Depends, Response, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncEngine

from agentic_rag import db
from agentic_rag.api.deps import get_engine, get_http_client
from agentic_rag.config import Settings, get_settings
from agentic_rag.providers import uses_ollama

logger = logging.getLogger(__name__)
router = APIRouter(tags=["ops"])

CheckStatus = Literal["ok", "error"]
HealthCheck = Callable[[], Awaitable[None]]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    checks: dict[str, CheckStatus]


def get_health_checks(
    engine: Annotated[AsyncEngine, Depends(get_engine)],
    http: Annotated[httpx.AsyncClient, Depends(get_http_client)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> dict[str, HealthCheck]:
    async def database() -> None:
        await db.ping(engine)

    async def ollama() -> None:
        resp = await http.get(f"{settings.ollama_base_url}/api/tags", timeout=5.0)
        resp.raise_for_status()

    async def llm_api() -> None:
        # Lists models: authenticates the key without spending tokens.
        key = settings.llm_api_key.get_secret_value() if settings.llm_api_key else ""
        resp = await http.get(
            f"{settings.llm_base_url.rstrip('/')}/models",
            headers={"Authorization": f"Bearer {key}"},
            timeout=5.0,
        )
        resp.raise_for_status()

    checks: dict[str, HealthCheck] = {"database": database}
    if uses_ollama(settings):
        checks["ollama"] = ollama
    if settings.llm_provider == "openai_compatible":
        checks["llm_api"] = llm_api
    return checks


@router.get("/health/live")
async def live() -> dict[str, str]:
    """Liveness: the process is up and serving. Unlike /health it checks no dependencies, so a
    container orchestrator does not restart the API just because Ollama is briefly down."""
    return {"status": "ok"}


@router.get("/health", response_model=HealthResponse)
async def health(
    response: Response,
    checks: Annotated[dict[str, HealthCheck], Depends(get_health_checks)],
) -> HealthResponse:
    results: dict[str, CheckStatus] = {}
    for name, check in checks.items():
        try:
            await check()
            results[name] = "ok"
        except Exception:
            logger.warning("health check failed: %s", name, exc_info=True)
            results[name] = "error"

    healthy = all(v == "ok" for v in results.values())
    if not healthy:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return HealthResponse(status="ok" if healthy else "degraded", checks=results)
