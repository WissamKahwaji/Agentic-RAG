# Agentic RAG

A retrieval-augmented generation (RAG) service that answers questions over your documents and
**cites the exact passages it used**. It starts as a classic retrieve-then-generate pipeline and
evolves into an agentic workflow that rewrites queries, grades retrieved evidence, retries when
evidence is weak, and checks its own answer before returning it. Every change to retrieval or
prompting is measured with an evaluation suite.

Runs fully locally on free, open models. Hosted model providers plug in behind the same interface.

> **Status:** milestones 1-2 complete: cited answers, chat UI, and measured hybrid retrieval
> with reranking. The agentic graph is next.
> See [Roadmap](#roadmap).

![Chat UI: an answer with numbered citation markers next to the evidence passages they point to](docs/screenshot.png)

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
- **AI:** Ollama (`qwen2.5:7b`, `bge-m3`), `bge-reranker-v2-m3` cross-encoder (sentence-transformers, Apple MPS), LangGraph *(planned)*
- **Data:** PostgreSQL 17: pgvector (HNSW) for semantic search, `tsvector` + GIN for keyword search
- **Frontend:** Next.js 16 (App Router), React 19, TypeScript, Tailwind CSS 4
- **Quality:** pytest, ruff, mypy (strict), Vitest, ESLint, Ragas / DeepEval *(planned)*

## Getting started

### Prerequisites

- [uv](https://docs.astral.sh/uv/) (Python 3.12 is installed automatically)
- Docker with Compose (Docker Desktop, Colima or OrbStack)
- [Ollama](https://ollama.com) running locally
- Node.js 22+
- GNU Make

### Run locally

```bash
# 1. Models
ollama pull qwen2.5:7b
ollama pull bge-m3

# 2. Dependencies and config
make setup            # installs Python and Node deps, creates .env from .env.example

# 3. Infrastructure and schema
make up               # Postgres + pgvector, waits until healthy
make migrate

# 4. Documents
make corpus           # optional: download a small public sample corpus into data/raw
make ingest           # or: make ingest DOCS=path/to/your/docs

# 5. API
make dev              # http://localhost:8000  ·  interactive docs at /docs

# 6. Chat UI (in another terminal)
make web              # http://localhost:3000

# or ask from the terminal
make ask Q="How do I create an HNSW index?"
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
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1200` / `200` | Chunk length and overlap, in characters |
| `RETRIEVAL_TOP_K` | `5` | Chunks passed to the model per question |
| `LLM_NUM_CTX` | `8192` | Model context window; Ollama's default is too small for RAG prompts |
| `RETRIEVAL_MODE` | `hybrid+rerank` | Retrieval pipeline: `dense`, `keyword`, `hybrid`, `dense+rerank`, `hybrid+rerank` |
| `RERANK_CANDIDATES` | `10` | First-stage results the reranker re-orders |

## Ingestion

```
file ──► load ──► sections ──► chunk ──► embed ──► Postgres
         PDF: one section per page          bge-m3     documents
         Markdown: one per heading path     (batched)  chunks + HNSW index
```

- **Supported formats:** PDF, Markdown, plain text.
- **Citation metadata:** chunks never cross a page or heading boundary, so each one cites exactly
  one location (`page 6` or `pgvector > HNSW > Index options`).
- **Contextual embeddings:** each chunk is embedded with its document title and heading path
  prepended, so short passages still retrieve well. The stored text stays unmodified for display.
- **Idempotent:** every document stores a fingerprint of its bytes *and* the ingestion recipe
  (pipeline version, embedding model, chunk size). Unchanged files are skipped; changed files, or
  a changed recipe, atomically replace that document's chunks.

## Retrieval

```
            ┌─► dense: bge-m3 embedding, cosine over HNSW ──┐
question ───┤                                               ├─► RRF fusion ─► top 10 ─► cross-encoder ─► top 5
            └─► keyword: Postgres full-text, ts_rank_cd ────┘                           rerank
```

- **Dense** search matches meaning ("make the index build faster" finds *Index Build Time*).
- **Keyword** search matches exact terms that embeddings blur (`hnsw.ef_search`,
  `vector_l2_ops`). Postgres ranking has no IDF, so code identifiers in the question are also
  matched as exact phrases and boosted.
- **Reciprocal Rank Fusion** merges the two lists by rank, not score, because cosine similarity
  and `ts_rank` live on unrelated scales.
- A **cross-encoder reranker** reads the question and each candidate passage together. It is
  much more accurate than comparing separate embeddings, and too slow for the whole corpus, so
  it only re-orders the top 10.

Every stage is measured against a labelled set of 34 questions (`make eval-retrieval`, details in
[`backend/evals/README.md`](backend/evals/README.md)):

| Pipeline | hit@1 | hit@5 | MRR@10 | p50 latency |
| --- | --- | --- | --- | --- |
| dense | 79% | 100% | 0.887 | 21 ms |
| keyword | 35% | 88% | 0.559 | 4 ms |
| hybrid | 76% | 97% | 0.868 | 25 ms |
| **hybrid + rerank** (default) | **82%** | **100%** | **0.907** | ~350 ms |

On this small, clean corpus, dense retrieval with a strong embedding model is already a high
baseline. Hybrid search mainly helps identifier-style questions, and reranking gives the best
first result. The pipeline is a setting (`RETRIEVAL_MODE`), so it can be re-chosen as the corpus
grows.

## Answering

```
question ──► embed ──► top-k cosine search (HNSW) ──► numbered sources ──► LLM ──► verify citations
```

- **Grounded by construction:** the prompt allows only the numbered sources and requires a
  citation on every claim. When the sources do not contain the answer, the model must say so.
- **Citations are verified, not trusted:** every `[n]` marker is mapped back to the retrieved
  chunk. Markers pointing at sources that were never provided are removed, and an answer is
  flagged `grounded: false` when it cites nothing or declines to answer.
- **Versioned prompts:** each response reports the prompt version (e.g. `grounded_answer@v2`), so
  quality changes can be traced to prompt changes. v2 raised the citation rate on code-heavy
  answers from 72% to 100% (18 runs over 6 questions).

```jsonc
// POST /ask {"question": "What is the difference between RAG-Sequence and RAG-Token?"}
{
  "answer": "RAG-Sequence uses the same retrieved document to generate the complete sequence [2]. RAG-Token can predict each target token based on a different document [2][4].",
  "grounded": true,
  "citations": [
    {"number": 2, "source": "papers/rag-lewis-2020.pdf", "location": "page 3", "score": 0.576, "content": "..."},
    {"number": 4, "source": "papers/rag-lewis-2020.pdf", "location": "page 3", "score": 0.573, "content": "..."}
  ],
  "model": "qwen2.5:7b",
  "prompt": "grounded_answer@v2",
  "timings": {"retrieval_ms": 162, "generation_ms": 3613}
}
```

## Chat UI

The UI is built for checking answers, not just reading them.

- Each claim in an answer carries a numbered marker. Hovering or focusing a marker highlights the
  exact passage it points to in the evidence column; clicking pins it.
- Evidence cards show the source file, location (page or heading path), relevance to the
  question, and the passage itself, rendered as Markdown for Markdown sources.
- Answers that cite nothing are flagged as unverified; questions the documents can't answer say so.
- The browser only talks to Next.js Route Handlers (`/api/ask`, `/api/documents`), which forward
  to the FastAPI service server-side: no CORS, no backend URL in the client, and failures are
  translated into actionable messages.

## API

| Endpoint | Description |
| --- | --- |
| `POST /ask` | Answer a question with verified citations |
| `GET /documents` | Ingested documents with chunk counts |
| `GET /health` | Database and model-server status (503 when degraded) |

## Development

```bash
make help        # list all targets
make check       # backend + frontend lint, type-check, unit tests (what CI runs)
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
│   │   ├── models.py        # ORM models (documents, chunks)
│   │   ├── embeddings.py    # Embedder protocol + Ollama implementation
│   │   ├── llm.py           # LLMProvider protocol + Ollama implementation
│   │   ├── retrieval.py     # dense, keyword and hybrid (RRF) retrievers
│   │   ├── reranking.py     # cross-encoder reranker stage
│   │   ├── search.py        # builds the pipeline from RETRIEVAL_MODE
│   │   ├── evals/           # retrieval metrics and eval runner
│   │   ├── prompts.py       # versioned prompt templates
│   │   ├── answering.py     # retrieve -> generate -> verify citations
│   │   ├── ingestion/       # loaders, chunking, pipeline, CLI
│   │   └── api/             # HTTP routers and dependencies
│   ├── evals/               # golden datasets and results
│   ├── migrations/          # Alembic migrations
│   └── tests/
├── scripts/                 # sample-corpus download
├── frontend/                # Next.js chat UI
│   └── src/
│       ├── app/             # page, layout, /api route handlers (proxy to FastAPI)
│       ├── components/      # chat, turn, evidence card, composer
│       └── lib/             # API types, citation-marker parsing
├── docker-compose.yml       # Postgres + pgvector
├── Makefile
└── .env.example
```

## Roadmap

- [x] **Foundation:** FastAPI service, typed config, Postgres + pgvector, migrations, health checks, CI-ready tooling
- [x] **Milestone 1: Cited RAG**
  - [x] Document ingestion: load, chunk, embed, store; idempotent re-ingest
  - [x] Vector retrieval and grounded answers with verified citations
  - [x] Chat UI with inspectable citations
- [x] **Milestone 2: Retrieval quality:** labelled retrieval eval, hybrid search (pgvector + Postgres full-text, reciprocal rank fusion) and a local cross-encoder reranker
- [ ] **Milestone 3: Agentic graph:** query rewriting, document grading, retry on weak evidence, answer self-check, all with bounded loops
- [ ] **Milestone 4: Evaluation:** Ragas / DeepEval suite with a versioned golden dataset; faithfulness, answer relevancy, context precision/recall
- [ ] **Milestone 5: Production:** containerized deploy, hosted LLM provider, tracing with self-hosted Langfuse

## License

MIT
