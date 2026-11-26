# Evaluation datasets

Run everything with `make eval` from the repo root. Reports land in `reports/` (committed, so
quality history is part of the repo). Pass `BASELINE=backend/evals/reports/<run>.json` to see
deltas against an earlier run.

## `golden.jsonl`

Labelled questions over the sample corpus (`make corpus`). The file is the versioned dataset:
every report records its SHA-256, so scores from different dataset versions are never compared
by accident.

| Field | Meaning |
| --- | --- |
| `id` | Stable identifier, used in reports of missed questions |
| `kind` | `keyword`: hinges on exact terms or identifiers (`hnsw.ef_search`). `natural`: paraphrased, shares few words with the source |
| `question` | The question as a user would ask it |
| `reference` | A correct, complete answer written from the source passages (ground truth for context recall, context precision and factual correctness) |
| `relevant` | Locations that answer it: `source` plus a heading-path prefix (`section`) or PDF `pages` |

References state only what the sources say, in plain sentences, without citations. They were
written by reading the labelled passages, and checked against the PDF text for numbers.

A retrieved chunk counts as relevant when its source matches and its heading path equals or
sits under a labelled `section`, or its page is in `pages`.

### Labelling rules

- A location is relevant if a reader could answer the question from that passage alone.
- Labels change only after reading the passage text, never because a pipeline ranks it highly.
  The 2026-11-26 audit added four locations (the sparse-vector operator table, the halfvec
  storage tip, binary quantization in the indexing tips, and the Jeopardy human-evaluation
  table on page 8). All four genuinely answer their question and had been missed when labelling.
  Every pipeline was re-scored after the fix. While writing reference answers, page 8 of the RAG
  paper (the index hot-swap experiment) was added to `rag-hot-swap`; retrieval scores did not
  change.
- 34 questions means one question is about 3 percentage points. Treat differences of one or two
  questions as noise.

## `unanswerable.jsonl`

12 questions close to the corpus topics that it cannot answer (CORS, WebSockets, DiskANN, the
learning rate used to train RAG, ...). A good pipeline declines them. Used by
`make eval-grading` and `make eval-answers`.

## Other evals

- `make eval-grading`: what each reranker threshold keeps or drops. Chose 0.05: keeps a relevant
  passage for 94% of answerable questions and drops about one irrelevant passage per question.
  No threshold separates "on topic" from "answerable" (the DiskANN question's passages score 0.89),
  so declining unanswerable questions is left to generation and self-check.
- `make eval-selfcheck`: self-check false alarms on real answers vs detection of a planted false
  claim. Per-claim checking: 18% false alarms, 100% detection (holistic: 59% / 100%).
- `make eval-answers`: simple vs agentic end to end; full answers saved to `evals/results/`
  (git-ignored) for review.

### Retrieval results (2026-11-26, sample corpus, 330 chunks)

| Pipeline | hit@1 | hit@5 | MRR@10 | p50 latency |
| --- | --- | --- | --- | --- |
| dense (bge-m3, HNSW) | 79% | 100% | 0.887 | 21 ms |
| keyword (Postgres FTS) | 35% | 88% | 0.559 | 4 ms |
| hybrid (RRF of both) | 76% | 97% | 0.868 | 25 ms |
| dense + rerank | 79% | 100% | 0.880 | ~350 ms |
| **hybrid + rerank** (default) | **82%** | **100%** | **0.907** | ~350 ms |

Reranking uses `BAAI/bge-reranker-v2-m3` over the top 10 candidates. 20 or 30 candidates gave
the same quality at 2-2.5x the latency.
