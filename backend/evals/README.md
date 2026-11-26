# Evaluation datasets

## `retrieval_golden.jsonl`

Labelled questions over the sample corpus (`make corpus`), used by `make eval-retrieval` to
compare retrieval pipelines.

| Field | Meaning |
| --- | --- |
| `id` | Stable identifier, used in reports of missed questions |
| `kind` | `keyword`: hinges on exact terms or identifiers (`hnsw.ef_search`). `natural`: paraphrased, shares few words with the source |
| `question` | The question as a user would ask it |
| `relevant` | Locations that answer it: `source` plus a heading-path prefix (`section`) or PDF `pages` |

A retrieved chunk counts as relevant when its source matches and its heading path equals or
sits under a labelled `section`, or its page is in `pages`.

### Labelling rules

- A location is relevant if a reader could answer the question from that passage alone.
- Labels change only after reading the passage text, never because a pipeline ranks it highly.
  The 2026-11-26 audit added four locations (the sparse-vector operator table, the halfvec
  storage tip, binary quantization in the indexing tips, and the Jeopardy human-evaluation
  table on page 8). All four genuinely answer their question and had been missed when labelling.
  Every pipeline was re-scored after the fix.
- 34 questions means one question is about 3 percentage points. Treat differences of one or two
  questions as noise.

### Results (2026-11-26, sample corpus, 330 chunks)

| Pipeline | hit@1 | hit@5 | MRR@10 | p50 latency |
| --- | --- | --- | --- | --- |
| dense (bge-m3, HNSW) | 79% | 100% | 0.887 | 21 ms |
| keyword (Postgres FTS) | 35% | 88% | 0.559 | 4 ms |
| hybrid (RRF of both) | 76% | 97% | 0.868 | 25 ms |
| dense + rerank | 79% | 100% | 0.880 | ~350 ms |
| **hybrid + rerank** (default) | **82%** | **100%** | **0.907** | ~350 ms |

Reranking uses `BAAI/bge-reranker-v2-m3` over the top 10 candidates. 20 or 30 candidates gave
the same quality at 2-2.5x the latency.
