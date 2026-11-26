# Agentic RAG

A retrieval-augmented generation (RAG) service that answers questions over your documents and
**cites the exact passages it used**. It starts as a classic retrieve-then-generate pipeline and
evolves into an agentic workflow that rewrites queries, grades retrieved evidence, retries when
evidence is weak, and checks its own answer before returning it. Every change to retrieval or
prompting is measured with an evaluation suite.

Runs fully locally on free, open models. Hosted model providers plug in behind the same interface.

> **Status:** early development. The project skeleton is in place; ingestion and cited answering
> are next. See [Roadmap](#roadmap).

---

## Why this design

| Concern | Decision | Rationale |
| --- | --- | --- |
| Trust | Answers must cite retrieved chunks; refuse when evidence is missing | Hallucination is the main failure mode of LLM apps; citations make answers verifiable |
| Retrieval quality | Hybrid search (vector + full-text) with a cross-encoder reranker | Dense vectors miss exact terms (IDs, names); keyword search misses paraphrases. Reranking fixes ordering |
| Control flow | LangGraph state machine with bounded loops | Explicit, inspectable agent steps instead of an opaque autonomous loop |
| Storage | Postgres + pgvector | One database for documents, vectors and full-text indexes; no extra vector DB to operate |
| Models | Ollama locally, Groq/Gemini hosted, behind one `LLMProvider` interface | No vendor lock-in; zero-cost local development; swap models without touching business logic |
| Quality | Ragas / DeepEval evaluation suite | Retrieval and prompt changes are justified with numbers, not impressions |

## Architecture

```mermaid
flowchart LR
    subgraph Ingestion
        D[Documents<br/>PDF / Markdown] --> L[Load & chunk]
        L --> E[Embed<br/>bge-m3]
        E --> PG[(Postgres<br/>pgvector + FTS)]
    end

    subgraph Query
        U[User / Next.js UI] --> API[FastAPI]
        API --> G[Agent graph<br/>LangGraph]
        G -->|rewrite, retrieve| PG
        G -->|grade, generate, self-check| LLM[LLM provider<br/>Ollama · Groq · Gemini]
        G --> API
        API -->|answer + citations| U
    end
```

## Tech stack

- **Backend:** Python 3.12, FastAPI, Pydantic v2, SQLAlchemy 2 (async), Alembic, uv
- **AI:** LangGraph, LangChain integrations, Ollama (`qwen2.5:7b`, `bge-m3`)
- **Data:** PostgreSQL 17 with pgvector (HNSW)
- **Frontend:** Next.js *(planned)*
- **Quality:** pytest, ruff, mypy (strict), Ragas / DeepEval *(planned)*

## Getting started

### Prerequisites

- [uv](https://docs.astral.sh/uv/) (Python 3.12 is installed automatically)
- Docker with Compose (Docker Desktop, Colima or OrbStack)
- [Ollama](https://ollama.com) running locally
- GNU Make

### Run locally

```bash
# 1. Models
ollama pull qwen2.5:7b
ollama pull bge-m3

# 2. Dependencies and config
make setup            # installs Python deps, creates .env from .env.example

# 3. Infrastructure and schema
make up               # Postgres + pgvector, waits until healthy
make migrate

# 4. API
make dev              # http://localhost:8000  ·  interactive docs at /docs
```

Check that everything is wired up:

```bash
curl localhost:8000/health
# {"status":"ok","checks":{"database":"ok","ollama":"ok"}}
```

### Configuration

All configuration is via environment variables (see [`.env.example`](.env.example)).

| Variable | Default | Purpose |
| --- | --- | --- |
| `DATABASE_URL` | `postgresql+psycopg://rag:rag@localhost:5432/rag` | Postgres connection |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server |
| `LLM_MODEL` | `qwen2.5:7b` | Generation model |
| `EMBEDDING_MODEL` | `bge-m3` | Embedding model |
| `EMBEDDING_DIM` | `1024` | Must match the embedding model and the vector column |

## Development

```bash
make help        # list all targets
make check       # lint + type-check + unit tests (what CI runs)
make test-all    # also runs integration tests against the live stack
make format      # auto-fix lint and formatting
```

Project layout:

```
.
├── backend/
│   ├── src/agentic_rag/     # application package
│   │   ├── main.py          # app factory and lifespan
│   │   ├── config.py        # typed settings
│   │   ├── db.py            # async database engine
│   │   └── api/             # HTTP routers and dependencies
│   ├── migrations/          # Alembic migrations
│   └── tests/
├── docker-compose.yml       # Postgres + pgvector
├── Makefile
└── .env.example
```

## Roadmap

- [x] **Foundation:** FastAPI service, typed config, Postgres + pgvector, migrations, health checks, CI-ready tooling
- [ ] **Milestone 1: Cited RAG:** document ingestion (load, chunk, embed), vector retrieval, grounded answers with citations, minimal chat UI
- [ ] **Milestone 2: Retrieval quality:** hybrid search (pgvector + Postgres full-text, reciprocal rank fusion) and a local cross-encoder reranker
- [ ] **Milestone 3: Agentic graph:** query rewriting, document grading, retry on weak evidence, answer self-check, all with bounded loops
- [ ] **Milestone 4: Evaluation:** Ragas / DeepEval suite with a versioned golden dataset; faithfulness, answer relevancy, context precision/recall
- [ ] **Milestone 5: Production:** containerized deploy, hosted LLM provider, tracing with self-hosted Langfuse

## License

MIT
