---
paths:
  - "backend/src/agentic_rag/**"
---

# Retrieval, generation and agent rules

- **Grounding is the product.** Every answer must be generated only from retrieved context and
  carry citations that map to real chunk ids. If the context does not support an answer, the
  system says so; it never fills gaps from model memory.
- **Models sit behind interfaces.** Application code depends on the `LLMProvider` /
  embedding protocols, never on a vendor SDK directly. Model names come from settings.
- **Embedding consistency.** The embedding model and `EMBEDDING_DIM` must match the vector
  column. Changing either requires a migration and a full re-index; call this out explicitly.
- **Re-index when the recipe changes.** Any change to loading, chunking or embedding logic must
  bump `PIPELINE_VERSION` in `ingestion/pipeline.py`, so the fingerprint forces re-ingestion.
- **Prompts are code.** Keep prompt templates in one module, named and versioned, with the
  variables they expect. No prompt strings scattered through business logic.
- **Structured output.** When the model must return fields (grades, rewritten queries,
  citations), request JSON and parse it into a Pydantic model; handle parse failure with a
  bounded retry and a safe fallback.
- **Bounded agents.** Every loop in the agent graph has a max-iterations limit and a defined
  exit. No unbounded retries.
- **Measure before tuning.** Changes to chunking, top-k, retrieval mode, reranking, or prompts
  must be justified with `make eval-retrieval` (and answer evals once they exist), with
  before/after numbers in the commit body.
- **Quality changes ship with a report.** Run `make eval BASELINE=<latest report>` before and
  after any change to prompts, retrieval, grading or the agent graph, commit the new report in
  `backend/evals/reports/`, and quote the deltas in the commit body. The judge model must not be
  the generator for scores that matter (the report flags it).
- **Golden labels follow the text.** Change a label in `backend/evals/` only after reading the
  passage and judging it answers the question. Never relabel to favour a pipeline; re-score
  every mode after any label change and note it in `backend/evals/README.md`.
- Log retrieval decisions (query, chunk ids, scores, latency) at DEBUG for traceability; never
  log full document contents at INFO.
