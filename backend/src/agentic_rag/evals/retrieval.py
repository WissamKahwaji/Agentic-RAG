"""Retrieval eval: `uv run python -m agentic_rag.evals.retrieval [--mode MODE ...] [-v]`.

Scores each retrieval mode against the golden set and prints hit@1, hit@5, MRR@10 and median
latency, overall and split by question kind.
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx

from agentic_rag import db
from agentic_rag.config import get_settings
from agentic_rag.embeddings import OllamaEmbedder
from agentic_rag.evals.metrics import GoldenQuestion, Summary, first_relevant_rank, summarize
from agentic_rag.reranking import CrossEncoderReranker
from agentic_rag.retrieval import Retriever
from agentic_rag.search import RETRIEVAL_MODES, build_retriever

GOLDEN = Path(__file__).resolve().parents[3] / "evals" / "golden.jsonl"


def load_golden(path: Path = GOLDEN) -> list[GoldenQuestion]:
    lines = path.read_text().splitlines()
    return [GoldenQuestion.model_validate(json.loads(line)) for line in lines if line.strip()]


async def evaluate(
    retriever: Retriever, golden: list[GoldenQuestion]
) -> tuple[dict[str, int | None], list[float]]:
    ranks: dict[str, int | None] = {}
    latencies: list[float] = []
    for q in golden:
        started = time.perf_counter()
        results = await retriever.search(q.question, k=10)
        latencies.append((time.perf_counter() - started) * 1000)
        ranks[q.id] = first_relevant_rank(q, results)
    return ranks, latencies


def _row(label: str, s: Summary, latency: bool = True) -> str:
    row = f"{label:<22} {s.n:>3}  {s.hit_at_1:>6.0%}  {s.hit_at_5:>6.0%}  {s.mrr_at_10:>6.3f}"
    return f"{row}  {s.p50_ms:>7.0f}" if latency else row


async def main(modes: list[str], verbose: bool) -> None:
    settings = get_settings()
    golden = load_golden()
    kinds = sorted({q.kind for q in golden})
    engine = db.create_engine(str(settings.database_url))
    try:
        async with httpx.AsyncClient(timeout=settings.http_timeout_s) as http:
            embedder = OllamaEmbedder(
                http, settings.ollama_base_url, settings.embedding_model, settings.embedding_dim
            )
            await embedder.embed(["warm-up"])  # exclude model load time from latency
            sessions = db.create_session_factory(engine)
            reranker = None
            if any("rerank" in m for m in modes):
                reranker = CrossEncoderReranker(settings.reranker_model)
                await reranker.warm_up()  # model download/load is not query latency

            print(
                f"{'mode':<22} {'n':>3}  {'hit@1':>6}  {'hit@5':>6}  {'MRR@10':>6}  {'p50 ms':>7}"
            )
            for mode in modes:
                retriever = build_retriever(mode, sessions, embedder, settings, reranker)
                await retriever.search("warm-up", k=1)
                ranks, latencies = await evaluate(retriever, golden)
                print(_row(mode, summarize(ranks, latencies)))
                for kind in kinds:
                    subset = {q.id: ranks[q.id] for q in golden if q.kind == kind}
                    print(_row(f"  {kind}", summarize(subset, []), latency=False))
                if verbose:
                    missed = summarize(ranks, latencies).misses
                    print(f"  missed @5: {', '.join(missed) or 'none'}")
    finally:
        await engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", action="append", choices=RETRIEVAL_MODES)
    parser.add_argument("-v", "--verbose", action="store_true", help="list questions missed @5")
    args = parser.parse_args()
    asyncio.run(main(args.mode or list(RETRIEVAL_MODES), args.verbose))
