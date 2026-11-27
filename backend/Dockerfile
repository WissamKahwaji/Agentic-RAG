# syntax=docker/dockerfile:1
# API image. Two stages: build the virtualenv with uv, then copy only what runs into a slim
# runtime image (no compilers, no uv, no build caches).

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS build
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never
WORKDIR /app
# Dependencies first, in their own layer: code changes do not invalidate it.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable


FROM python:3.12-slim-bookworm AS runtime
RUN groupadd --system app && useradd --system --gid app --create-home --home-dir /home/app app
WORKDIR /app
COPY --from=build --chown=app:app /app/.venv /app/.venv
COPY --chown=app:app alembic.ini ./
COPY --chown=app:app migrations ./migrations
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    # Model downloads (the reranker) go to a volume, not into the image.
    HF_HOME=/home/app/.cache/huggingface
USER app
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health/live', timeout=2)"
CMD ["uvicorn", "agentic_rag.main:app", "--host", "0.0.0.0", "--port", "8000"]
