# Agentic RAG

Retrieval-augmented generation service that answers questions over a document corpus with
verifiable citations. Built to grow from a single retrieve-then-generate pass into an agentic
graph (query rewriting, document grading, retries, answer self-check) with measured quality.

## Stack

| Layer        | Choice                                                                |
| ------------ | --------------------------------------------------------------------- |
| API          | Python 3.12, FastAPI, Pydantic v2, uv                                 |
| Agent flow   | LangGraph (plus LangChain loaders/integrations where they save code)  |
| LLM          | Ollama locally (`qwen2.5:7b`); hosted Groq/Gemini behind one interface |
| Embeddings   | `bge-m3` via Ollama (1024 dims)                                       |
| Storage      | Postgres 17 + pgvector, SQLAlchemy 2 async + psycopg 3, Alembic       |
| Frontend     | Next.js (`frontend/`, not yet created)                                |
| Quality      | pytest, Ragas/DeepEval evals, ruff, mypy --strict                     |

Ollama runs **natively on the host** (Metal GPU). Only Postgres runs in Docker.

## Layout

```
backend/src/agentic_rag/   application package
  main.py                  app factory + lifespan (engine, http client on app.state)
  config.py                Settings (pydantic-settings, reads repo-root .env)
  db.py                    async engine helpers
  models.py                ORM: Document, Chunk (vector(1024), HNSW cosine)
  embeddings.py            Embedder protocol + OllamaEmbedder
  ingestion/               loaders -> chunking -> pipeline (+ `python -m` CLI)
  api/                     routers + FastAPI dependencies
backend/migrations/        Alembic (revision ids are sequential: 0001, 0002, ...)
backend/tests/             pytest; `integration` marker = needs live stack
docker-compose.yml         Postgres + pgvector
```

## Commands (run from repo root)

```bash
make setup      # uv sync + create .env
make up         # start Postgres, wait for healthy
make migrate    # alembic upgrade head
make corpus     # download public sample docs into data/raw (git-ignored)
make ingest     # ingest data/raw (or DOCS=path)
make dev        # API on :8000 with reload; docs at /docs
make check      # lint + typecheck + unit tests  <- must pass before any task is "done"
make test-all   # includes integration tests (needs `make up` + Ollama)
```

Single test: `cd backend && uv run pytest tests/test_health.py::test_name -x`.

## Working agreements

- **Verify before claiming done.** Run `make check`; run `make test-all` when touching DB,
  retrieval, or model calls. Report failures verbatim, never paraphrase them away.
- **Small, reviewable steps.** One concern per change. Explain the design trade-off of any
  non-obvious choice in the PR/commit body, not in code comments.
- **No new dependency without a reason.** Prefer the standard library, then what is already in
  `pyproject.toml`. Add with `uv add` (never edit the lockfile by hand).
- **Configuration is environment-driven.** New settings go in `config.Settings` and
  `.env.example` together. Never hardcode URLs, model names, or credentials.
- **Never commit secrets.** `.env` is git-ignored; do not read or print it.
- **Git:** conventional commits (`feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:`).
  Do not push, create remotes, or open PRs unless explicitly asked.

Topic-specific rules live in `.claude/rules/` and load automatically for matching paths.
