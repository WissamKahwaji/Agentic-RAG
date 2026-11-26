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
- **Prompts are code.** Keep prompt templates in one module, named and versioned, with the
  variables they expect. No prompt strings scattered through business logic.
- **Structured output.** When the model must return fields (grades, rewritten queries,
  citations), request JSON and parse it into a Pydantic model; handle parse failure with a
  bounded retry and a safe fallback.
- **Bounded agents.** Every loop in the agent graph has a max-iterations limit and a defined
  exit. No unbounded retries.
- **Measure before tuning.** Changes to chunking, top-k, reranking, or prompts should be
  justified by the eval suite (once it exists), with before/after numbers in the commit body.
- Log retrieval decisions (query, chunk ids, scores, latency) at DEBUG for traceability; never
  log full document contents at INFO.
