---
paths:
  - "backend/**/*.py"
---

# Python conventions

- Python 3.12 syntax: `X | None`, built-in generics (`list[str]`), `type` aliases where helpful.
- Full type hints everywhere; `mypy --strict` must pass. No `# type: ignore` without a
  specific error code and a one-line reason.
- Async all the way through request paths (SQLAlchemy async, `httpx.AsyncClient`). Never call
  blocking I/O (`requests`, sync DB drivers, `time.sleep`) inside an async function; if a
  library is sync-only (e.g. some loaders), run it with `asyncio.to_thread`.
- Shared clients (DB engine, HTTP client, model clients) are created once in the app lifespan
  and injected with FastAPI `Depends`. Do not create clients per request or at import time.
- Pydantic models for every request/response body and for structured LLM output. Validate at
  the boundary; pass typed objects inward, not dicts.
- Errors: raise specific exceptions; map them to HTTP errors in the API layer only. Never
  swallow exceptions silently. Log with `logger = logging.getLogger(__name__)`, no `print`.
- Keep modules focused. Business logic does not import from `api/`; `api/` is a thin layer.
- Docstrings explain *why* or contracts, not what the next line does.
