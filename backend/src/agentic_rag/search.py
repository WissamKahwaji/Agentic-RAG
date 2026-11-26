"""Assemble a retrieval pipeline from settings. Modes are named after their stages."""

from typing import get_args

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from agentic_rag.config import RetrievalMode, Settings
from agentic_rag.embeddings import Embedder
from agentic_rag.reranking import Reranker, RerankingRetriever
from agentic_rag.retrieval import HybridRetriever, KeywordRetriever, Retriever, VectorRetriever

RETRIEVAL_MODES: tuple[RetrievalMode, ...] = get_args(RetrievalMode)


def build_retriever(
    mode: str,
    session_factory: async_sessionmaker[AsyncSession],
    embedder: Embedder,
    settings: Settings,
    reranker: Reranker | None = None,
) -> Retriever:
    if mode not in RETRIEVAL_MODES:
        raise ValueError(f"unknown retrieval mode {mode!r}; expected one of {RETRIEVAL_MODES}")

    first_stage, _, rerank = mode.partition("+")
    dense = VectorRetriever(session_factory, embedder)
    retriever: Retriever
    if first_stage == "dense":
        retriever = dense
    elif first_stage == "keyword":
        retriever = KeywordRetriever(session_factory)
    else:
        retriever = HybridRetriever(
            [dense, KeywordRetriever(session_factory)], candidates=settings.hybrid_candidates
        )

    if rerank:
        if reranker is None:
            raise ValueError(f"mode {mode!r} needs a reranker")
        retriever = RerankingRetriever(retriever, reranker, candidates=settings.rerank_candidates)
    return retriever
