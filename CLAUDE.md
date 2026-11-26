# Agentic RAG

Retrieval-augmented generation service that answers questions over a document corpus with
verifiable citations. Built to grow from a single retrieve-then-generate pass into an agentic
graph (query rewriting, document grading, retries, answer self-check) with measured quality.

## Stack

| Layer        | Choice                                                                |
| ------------ | --------------------------------------------------------------------- |
| API          | Python 3.12, FastAPI, Pydantic v2, uv                                 |
| Agent flow   | LangGraph state graph (`agent/`); LangChain only for text splitters   |
| LLM          | Ollama locally (`qwen2.5:7b`); hosted Groq/Gemini behind one interface |
| Embeddings   | `bge-m3` via Ollama (1024 dims)                                       |
| Reranker     | `BAAI/bge-reranker-v2-m3` cross-encoder, sentence-transformers on MPS |
| Storage      | Postgres 17 + pgvector, SQLAlchemy 2 async + psycopg 3, Alembic       |
| Frontend     | Next.js 16 App Router, React 19, TS, Tailwind 4 (`frontend/`)         |
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
  llm.py                   LLMProvider protocol + OllamaProvider (sets num_ctx explicitly)
  retrieval.py             Vector / Keyword (tsvector) / Hybrid (RRF) retrievers
  reranking.py             CrossEncoderReranker + RerankingRetriever
  search.py                build_retriever(RETRIEVAL_MODE) composes the stages
  evals/                   retrieval metrics + runner (golden set in backend/evals/)
  prompts.py               versioned prompts; bump version on any wording change
  answering.py             Answer models + simple AnswerService (retrieve -> generate)
  agent/graph.py           LangGraph agent: condense, retrieve, grade, reformulate, refuse,
                           generate, self_check; every loop bounded by settings
  agent/verify.py          per-claim citation verification (the self-check)
  ask_cli.py               `make ask` client for the running API
  ingestion/               loaders -> chunking -> pipeline (+ `python -m` CLI)
  api/                     routers + FastAPI dependencies
backend/migrations/        Alembic (revision ids are sequential: 0001, 0002, ...)
backend/tests/             pytest; `integration` marker = needs live stack
frontend/src/app/          page + /api/* Route Handlers that proxy to FastAPI
frontend/src/components/   chat UI (client components)
frontend/src/lib/          API types (mirror backend Pydantic models), citation parsing
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
make ask Q="…"  # call POST /ask on the running API
make eval-retrieval  # hit@1 / hit@5 / MRR per retrieval mode on the golden set
make eval-grading    # relevance-threshold calibration
make eval-selfcheck  # self-check false alarms vs planted false claims
make eval-answers    # simple vs agentic end to end (~10 min)
make web        # chat UI on :3000 (needs `make dev`)
make check      # backend + frontend lint, types, unit tests  <- must pass before "done"
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
