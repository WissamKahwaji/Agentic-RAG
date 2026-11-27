"""Calibrate answer self-checks: `uv run python -m agentic_rag.evals.selfcheck`.

For each golden question: retrieve, generate an answer, then build a *perturbed* copy with one
planted false claim that cites a real source. A good checker flags the perturbed answers
(detection rate) and leaves the originals alone (false-alarm rate). Originals can contain real
mistakes, so the false-alarm rate is an upper bound; review flagged originals by hand.
"""

import asyncio
import re

import httpx
from pydantic import BaseModel, Field

from agentic_rag import db
from agentic_rag.agent.verify import unsupported_claims
from agentic_rag.answering import is_refusal
from agentic_rag.config import get_settings
from agentic_rag.embeddings import OllamaEmbedder
from agentic_rag.evals.retrieval import load_golden
from agentic_rag.llm import ChatMessage, LLMProvider, OllamaProvider, chat_structured
from agentic_rag.prompts import (
    GROUNDED_ANSWER,
    SELF_CHECK,
    grounded_answer_user_message,
    self_check_user_message,
)
from agentic_rag.reranking import CrossEncoderReranker
from agentic_rag.retrieval import RetrievedChunk
from agentic_rag.search import build_retriever

# Plausible but false for this corpus, phrased like real documentation claims.
PLANTED = {
    "pgvector": "pgvector can also build HNSW indexes on the GPU with CUDA for faster builds",
    "fastapi": "FastAPI requires every path operation function to be declared with async def",
    "papers": "RAG was trained on 1,024 TPU v4 chips for three months before evaluation",
}


class _Verdict(BaseModel):
    supported: bool
    unsupported_claims: list[str] = Field(default_factory=list)


async def holistic_flags(llm: LLMProvider, answer: str, chunks: list[RetrievedChunk]) -> bool:
    """Self-check v1: one call that audits the whole answer."""
    verdict = await chat_structured(
        llm,
        [
            ChatMessage("system", SELF_CHECK.system),
            ChatMessage("user", self_check_user_message(answer, chunks)),
        ],
        _Verdict,
    )
    return not verdict.supported or bool([c for c in verdict.unsupported_claims if c.strip()])


async def per_claim_flags(llm: LLMProvider, answer: str, chunks: list[RetrievedChunk]) -> bool:
    """Self-check v2: one yes/no call per cited claim."""
    return bool(await unsupported_claims(llm, answer, chunks))


def plant(answer: str, chunks: list[RetrievedChunk]) -> str:
    cited = [int(n) for n in re.findall(r"\[(\d+)\]", answer) if 1 <= int(n) <= len(chunks)]
    n = cited[0] if cited else 1
    domain = chunks[n - 1].source.split("/")[0]
    return f"{answer}\n\n{PLANTED[domain]} [{n}]."


async def main() -> None:
    settings = get_settings()
    engine = db.create_engine(str(settings.database_url))
    checkers = {"holistic (v1)": holistic_flags, "per-claim (v2)": per_claim_flags}
    flags: dict[str, dict[str, list[bool]]] = {
        name: {"original": [], "planted": []} for name in checkers
    }
    flagged_originals: dict[str, list[str]] = {name: [] for name in checkers}
    try:
        async with httpx.AsyncClient(timeout=settings.http_timeout_s) as http:
            embedder = OllamaEmbedder(
                http, settings.ollama_base_url, settings.embedding_model, settings.embedding_dim
            )
            llm = OllamaProvider(
                http, settings.ollama_base_url, settings.llm_model, settings.llm_num_ctx
            )
            reranker = CrossEncoderReranker(
                settings.reranker_model, device=settings.reranker_device
            )
            retriever = build_retriever(
                "hybrid+rerank", db.create_session_factory(engine), embedder, settings, reranker
            )
            for q in load_golden():
                chunks = await retriever.search(q.question, settings.retrieval_top_k)
                result = await llm.chat(
                    [
                        ChatMessage("system", GROUNDED_ANSWER.system),
                        ChatMessage("user", grounded_answer_user_message(q.question, chunks)),
                    ],
                    temperature=settings.llm_temperature,
                )
                answer = result.text.strip()
                if is_refusal(answer):
                    continue
                for name, checker in checkers.items():
                    original = await checker(llm, answer, chunks)
                    planted = await checker(llm, plant(answer, chunks), chunks)
                    flags[name]["original"].append(original)
                    flags[name]["planted"].append(planted)
                    if original:
                        flagged_originals[name].append(q.id)
                print(f"  {q.id}")
    finally:
        await engine.dispose()

    print(f"\n{'checker':<16} {'false alarms (originals)':>26} {'detected (planted)':>20}")
    for name, f in flags.items():
        n = len(f["original"])
        print(
            f"{name:<16} {sum(f['original']):>17}/{n} ({sum(f['original']) / n:.0%})"
            f" {sum(f['planted']):>12}/{n} ({sum(f['planted']) / n:.0%})"
        )
    for name, ids in flagged_originals.items():
        print(f"  {name} flagged originals: {', '.join(ids) or 'none'}")


if __name__ == "__main__":
    asyncio.run(main())
