"""End-to-end answer eval: `uv run python -m agentic_rag.evals.answers [--mode simple|agentic]`.

Runs every answerable (golden) and unanswerable question through each answer pipeline and
reports behaviour, not wording:

- answered:   answerable questions that got a non-refusal answer (higher is better)
- grounded:   answerable questions whose answer cites sources and, for the agent, passed the
              self-check
- refused:    unanswerable questions correctly declined (higher is better); every other
              unanswerable answer is a likely hallucination
- latency:    median and p90 seconds per question
Full answers are written to backend/evals/results/ for manual review.
"""

import argparse
import asyncio
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from statistics import median, quantiles

import httpx

from agentic_rag import db
from agentic_rag.answering import Answer
from agentic_rag.config import get_settings
from agentic_rag.embeddings import OllamaEmbedder
from agentic_rag.evals.grading import load_unanswerable
from agentic_rag.evals.retrieval import load_golden
from agentic_rag.llm import OllamaProvider
from agentic_rag.main import build_answer_service
from agentic_rag.reranking import CrossEncoderReranker

RESULTS = Path(__file__).resolve().parents[3] / "evals" / "results"


async def run_mode(mode: str, reranker: CrossEncoderReranker) -> list[dict[str, object]]:
    settings = get_settings().model_copy(update={"answer_mode": mode})
    engine = db.create_engine(str(settings.database_url))
    rows: list[dict[str, object]] = []
    try:
        async with httpx.AsyncClient(timeout=settings.http_timeout_s) as http:
            embedder = OllamaEmbedder(
                http, settings.ollama_base_url, settings.embedding_model, settings.embedding_dim
            )
            llm = OllamaProvider(
                http, settings.ollama_base_url, settings.llm_model, settings.llm_num_ctx
            )
            service = await build_answer_service(
                settings, db.create_session_factory(engine), embedder, llm, http, reranker
            )
            questions = [(q.id, q.question, True) for q in load_golden()] + [
                (qid, question, False) for qid, question in load_unanswerable()
            ]
            for qid, question, answerable in questions:
                started = time.perf_counter()
                answer: Answer = await service.answer(question)
                rows.append(
                    {
                        "id": qid,
                        "question": question,
                        "answerable": answerable,
                        "seconds": round(time.perf_counter() - started, 2),
                        **answer.model_dump(mode="json", exclude={"citations"}),
                        "cited_sources": [f"{c.source} ({c.location})" for c in answer.citations],
                    }
                )
                print(f"  {mode:<8} {qid:<26} {'refused' if answer.refused else 'answered'}")
    finally:
        await engine.dispose()
    return rows


def report(mode: str, rows: list[dict[str, object]]) -> None:
    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    secs = sorted(float(r["seconds"]) for r in rows)  # type: ignore[arg-type]
    answered = sum(not r["refused"] for r in answerable)
    grounded = sum(bool(r["grounded"]) for r in answerable)
    refused = sum(bool(r["refused"]) for r in unanswerable)
    p90 = quantiles(secs, n=10)[-1] if len(secs) >= 2 else secs[0]
    print(
        f"{mode:<8}  answered {answered}/{len(answerable)}  grounded {grounded}/{len(answerable)}"
        f"  refused {refused}/{len(unanswerable)}  latency p50 {median(secs):.1f}s p90 {p90:.1f}s"
    )
    for r in unanswerable:
        if not r["refused"]:
            print(f"    answered unanswerable {r['id']}: {str(r['answer'])[:140]!r}")
    for r in answerable:
        if r["refused"]:
            print(f"    refused answerable {r['id']}")


async def main(modes: list[str]) -> None:
    settings = get_settings()
    reranker = CrossEncoderReranker(settings.reranker_model)
    await reranker.warm_up()
    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    results = {mode: await run_mode(mode, reranker) for mode in modes}
    print()
    for mode, rows in results.items():
        (RESULTS / f"answers-{mode}-{stamp}.json").write_text(json.dumps(rows, indent=2))
        report(mode, rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", action="append", choices=["simple", "agentic"])
    args = parser.parse_args()
    asyncio.run(main(args.mode or ["simple", "agentic"]))
