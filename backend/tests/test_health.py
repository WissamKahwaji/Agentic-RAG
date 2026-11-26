import httpx
import pytest
from fastapi import FastAPI

from agentic_rag.api.health import HealthCheck, get_health_checks


async def _ok() -> None:
    return None


async def _fail() -> None:
    raise ConnectionError("down")


def _override(app: FastAPI, checks: dict[str, HealthCheck]) -> None:
    app.dependency_overrides[get_health_checks] = lambda: checks


async def test_health_ok_when_all_dependencies_up(app: FastAPI, client: httpx.AsyncClient) -> None:
    _override(app, {"database": _ok, "ollama": _ok})

    resp = await client.get("/health")

    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "checks": {"database": "ok", "ollama": "ok"}}


async def test_health_degraded_returns_503(app: FastAPI, client: httpx.AsyncClient) -> None:
    _override(app, {"database": _ok, "ollama": _fail})

    resp = await client.get("/health")

    assert resp.status_code == 503
    assert resp.json()["checks"] == {"database": "ok", "ollama": "error"}


@pytest.mark.integration
async def test_health_against_real_stack(client: httpx.AsyncClient) -> None:
    resp = await client.get("/health")

    assert resp.status_code == 200, resp.json()
