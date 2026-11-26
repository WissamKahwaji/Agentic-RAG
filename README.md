# Agentic RAG

A retrieval-augmented generation (RAG) service that answers questions over your documents and
**cites the exact passages it used**, then verifies each cited claim before answering. It starts as a classic retrieve-then-generate pipeline and
evolves into an agentic workflow that rewrites queries, grades retrieved evidence, retries when
evidence is weak, and checks its own answer before returning it. Every change to retrieval or
prompting is measured with an evaluation suite.

Runs fully locally on free, open models. Hosted model providers plug in behind the same interface.

> **Status:** milestones 1-4 complete: cited answers, chat UI, measured hybrid retrieval with
> reranking, an agentic graph with self-checking, and an evaluation suite. Deployment is next.
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
| Quality | Ragas evaluation suite on a versioned dataset | Retrieval and prompt changes are justified with numbers, not impressions |

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
- **AI:** LangGraph, Ollama (`qwen2.5:7b`, `bge-m3`), `bge-reranker-v2-m3` cross-encoder (sentence-transformers, Apple MPS)
- **Data:** PostgreSQL 17: pgvector (HNSW) for semantic search, `tsvector` + GIN for keyword search
- **Frontend:** Next.js 16 (App Router), React 19, TypeScript, Tailwind CSS 4
- **Quality:** pytest, ruff, mypy (strict), Vitest, ESLint, Ragas

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
| `ANSWER_MODE` | `agentic` | `agentic` (LangGraph agent) or `simple` (one retrieve-generate pass) |
| `RELEVANCE_THRESHOLD` | `0.05` | Reranker score a passage needs to reach the answer step |
| `SELF_CHECK` | `true` | Verify cited claims after generating |

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

## Agent

With `ANSWER_MODE=agentic` (the default), answering runs as a LangGraph state machine:

```mermaid
flowchart LR
    S([question + history]) --> C[condense<br/>follow-up → standalone query]
    C --> R[retrieve<br/>hybrid, 10 candidates]
    R --> G[grade<br/>cross-encoder ≥ 0.05]
    G -- relevant passages --> W[generate<br/>grounded, cited]
    G -- none, retries left --> F[reformulate<br/>new query] --> R
    G -- none, no retries --> X([decline])
    W --> V[self-check<br/>verify each cited claim]
    V -- unsupported, attempts left --> W
    V --> E([answer + trace])
```

- **Condense** turns "What about IVFFlat?" into "How do I create an IVFFlat index?" using the
  conversation, so follow-up questions retrieve the right passages.
- **Grade** keeps passages the cross-encoder scores at or above 0.05, a threshold calibrated on
  labelled questions (`make eval-grading`). If none pass, the agent **reformulates** the query and
  searches again, then **declines** without calling the generator.
- **Self-check** splits the answer into cited claims and asks one narrow yes/no question per claim,
  showing only the sources that claim cites. Unsupported claims trigger one revision. If they
  survive it, the answer is returned with those claims flagged.
- Every loop is bounded by settings (`MAX_RETRIEVAL_ATTEMPTS`, `MAX_GENERATION_ATTEMPTS`), and
  every response includes a `trace` of the steps taken, shown in the UI.

![Follow-up question with the rewritten query, a flagged mis-citation, and the step trace](docs/agent-trace.png)

### What the measurements showed

**Self-check design** (`make eval-selfcheck`). Each real answer is checked as-is and with one
planted false claim:

| Self-check | false alarms on real answers | planted false claims caught |
| --- | --- | --- |
| holistic: "list every unsupported claim" | 59% | 100% |
| **per-claim yes/no** (used) | **18%** | **100%** |

A 7B model is a poor auditor of a whole answer but a reasonable judge of one claim against one
passage.

**Simple vs agentic, end to end** (`make eval-answers`, 34 answerable + 12 unanswerable questions):

| Pipeline | answered | correctly declined | p50 / p90 latency |
| --- | --- | --- | --- |
| simple (retrieve → generate) | 34 / 34 | 12 / 12 | 1.5 s / 3.9 s |
| agentic | 31 / 34 | 12 / 12 | 1.6 s / 4.6 s |

On this small, single-topic corpus, the simple pipeline already declines well, so the agent does
not raise headline numbers. Its value shows in the cases behind them:

- One of the agent's three declines replaced a **confidently wrong** simple answer. The simple
  pipeline misreported RAG's FEVER accuracy; the self-check caught the unsupported numbers and
  the agent declined instead.
- A retry with a reformulated query rescued a question where the first search found nothing
  relevant.
- Follow-up questions work at all only in the agentic pipeline (not in the table above).
- Answers that survive with unsupported claims are flagged instead of presented as verified.

The cost is a longer tail: a revision means a second generation pass.

## Evaluation

`make eval` scores the whole system on a versioned dataset (34 answerable questions with
reference answers written from the sources, 12 unanswerable ones) and writes a report with the
git commit, dataset hash, models and prompt versions ([`backend/evals/reports/`](backend/evals/reports/)).
Quality metrics come from [Ragas](https://docs.ragas.io), with an LLM as the judge.

Baseline (2026-11-26):

| metric | simple | agentic |
| --- | --- | --- |
| answered (answerable) | 100% | 94% |
| declined (unanswerable) | 100% | 100% |
| faithfulness | 0.942 | 0.925 |
| answer relevancy | 0.862 | 0.732 |
| context precision | 0.822 | 0.805 |
| context recall | 0.865 | 0.821 |
| factual correctness | 0.545 | 0.504 |
| latency p50 / p90 | 3.3 s / 5.4 s | 3.9 s / 7.6 s |

How to read it:

- **The judge is the generator (qwen2.5:7b), so absolute scores are biased upward and noisy.**
  Two near-identical answers received relevancy scores 0.29 apart, and a correct answer about
  vector dimensions scored 0 on factual correctness. Differences smaller than about 0.05 should
  not be trusted. A stronger, independent judge is a configuration change (`EVAL_JUDGE_*`).
- Part of the agent's lower relevancy and correctness comes from its two declines, which score 0
  on those metrics. One of them replaced a confidently wrong answer (see [Agent](#agent)).
- Both pipelines fail `rag-generator` ("which generator model does RAG use?"): the passage naming
  BART-large is retrieved but not used. That is the next retrieval/generation issue to fix.

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
│   │   ├── answering.py     # answer models, citation verification, simple pipeline
│   │   ├── agent/           # LangGraph agent graph and per-claim self-check
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
- [x] **Milestone 3: Agentic graph:** follow-up condensing, calibrated passage grading, query reformulation, per-claim self-check, bounded loops, step trace
- [x] **Milestone 4: Evaluation:** Ragas suite on a versioned dataset with reference answers; faithfulness, answer relevancy, context precision/recall, factual correctness; reports with config snapshots and baseline deltas
- [ ] **Milestone 5: Production:** containerized deploy, hosted LLM provider, tracing with self-hosted Langfuse

## License

MIT
