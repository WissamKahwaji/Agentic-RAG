"""Calibrate the relevance threshold used to grade passages.

`uv run python -m agentic_rag.evals.grading`

For every golden question (answerable) and every unanswerable question, take the first-stage
candidates, score them with the reranker, and report what each threshold would keep:

- answerable kept relevant: questions that keep at least one labelled-relevant passage
- unanswerable refused:     questions where nothing passes, so the agent declines to answer
- irrelevant kept / q:      passages kept that are not labelled relevant (noise sent to the LLM)
"""

import asyncio
import json
from pathlib import Path

import httpx

from agentic_rag import db
from agentic_rag.config import get_settings
from agentic_rag.embeddings import OllamaEmbedder
from agentic_rag.evals.metrics import GoldenQuestion
from agentic_rag.evals.retrieval import load_golden
from agentic_rag.reranking import CrossEncoderReranker, RerankingRetriever
from agentic_rag.search import build_retriever

UNANSWERABLE = Path(__file__).resolve().parents[3] / "evals" / "unanswerable.jsonl"
THRESHOLDS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5)


def load_unanswerable(path: Path = UNANSWERABLE) -> list[tuple[str, str]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [(r["id"], r["question"]) for r in rows]


async def main() -> None:
    settings = get_settings()
    engine = db.create_engine(str(settings.database_url))
    try:
        async with httpx.AsyncClient(timeout=settings.http_timeout_s) as http:
            embedder = OllamaEmbedder(
                http, settings.ollama_base_url, settings.embedding_model, settings.embedding_dim
            )
            reranker = CrossEncoderReranker(settings.reranker_model)
            await reranker.warm_up()
            first_stage = build_retriever(
                "hybrid", db.create_session_factory(engine), embedder, settings
            )
            retriever = RerankingRetriever(first_stage, reranker, settings.rerank_candidates)
            k = settings.rerank_candidates

            answerable: list[tuple[GoldenQuestion, list[tuple[float, bool]]]] = []
            for q in load_golden():
                results = await retriever.search(q.question, k)
                labelled = [(r.score, any(s.matches(r) for s in q.relevant)) for r in results]
                answerable.append((q, labelled))

            unanswerable: list[tuple[str, list[float]]] = []
            for qid, question in load_unanswerable():
                results = await retriever.search(question, k)
                unanswerable.append((qid, [r.score for r in results]))
    finally:
        await engine.dispose()

    n_a, n_u = len(answerable), len(unanswerable)
    print(f"{n_a} answerable, {n_u} unanswerable questions; top {k} reranked candidates each\n")
    print(
        f"{'threshold':>9}  {'answerable kept relevant':>24}  {'unanswerable refused':>20}  "
        f"{'irrelevant kept/q':>17}"
    )
    for t in THRESHOLDS:
        kept_rel = sum(any(s >= t and rel for s, rel in c) for _, c in answerable)
        refused = sum(all(s < t for s in scores) for _, scores in unanswerable)
        noise = sum(sum(s >= t and not rel for s, rel in c) for _, c in answerable) / n_a
        print(f"{t:>9.2f}  {kept_rel / n_a:>24.0%}  {refused / n_u:>20.0%}  {noise:>17.2f}")

    print("\nhighest score among unanswerable questions:")
    for qid, scores in sorted(unanswerable, key=lambda x: -max(x[1], default=0)):
        print(f"  {max(scores, default=0):.3f}  {qid}")
    print("\nlowest best-relevant score among answerable questions:")
    best = [(max((s for s, rel in c if rel), default=0.0), q.id) for q, c in answerable]
    for score, qid in sorted(best)[:6]:
        print(f"  {score:.3f}  {qid}")


if __name__ == "__main__":
    asyncio.run(main())
