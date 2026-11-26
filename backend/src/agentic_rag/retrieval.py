"""Retrieval over stored chunks: dense (vector), keyword (full-text), and their fusion.

Every retriever implements the same `Retriever` protocol, so stages compose:
    HybridRetriever(dense, keyword) -> RerankingRetriever(hybrid, reranker)
"""

import asyncio
import logging
import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, Protocol

from sqlalchemy import (
    ColumnElement,
    Float,
    Integer,
    Select,
    cast,
    func,
    literal_column,
    select,
)
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from agentic_rag.embeddings import Embedder
from agentic_rag.models import Chunk, Document

logger = logging.getLogger(__name__)

# Tokens joined by `_` or `.` (snake_case names, dotted settings): `hnsw.ef_search`.
_IDENTIFIER = re.compile(r"\b[A-Za-z0-9]+(?:[._][A-Za-z0-9]+)+\b")


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: uuid.UUID
    source: str
    title: str
    page: int | None
    section: str | None
    content: str
    # Relevance from the stage that produced this result; higher is better. Its scale depends
    # on the stage (cosine similarity, ts_rank, fused rank, reranker probability).
    score: float

    @property
    def location(self) -> str:
        """Human-readable citation location, e.g. `page 6` or `pgvector > HNSW`."""
        if self.page is not None:
            return f"page {self.page}"
        return self.section or self.title


class Retriever(Protocol):
    async def search(self, query: str, k: int) -> list[RetrievedChunk]: ...


def _select_chunks(score: Any) -> Select[Any]:
    return select(Chunk, Document.source, Document.title, score.label("score")).join(
        Document, Document.id == Chunk.document_id
    )


def _to_results(rows: Sequence[Any]) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk_id=chunk.id,
            source=source,
            title=title,
            page=chunk.page,
            section=chunk.section,
            content=chunk.content,
            score=float(score),
        )
        for chunk, source, title, score in rows
    ]


class VectorRetriever:
    """Dense retrieval: nearest chunks by cosine similarity of embeddings (semantic match)."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession], embedder: Embedder):
        self._sessions = session_factory
        self._embedder = embedder

    async def search(self, query: str, k: int) -> list[RetrievedChunk]:
        [query_vector] = await self._embedder.embed([query])
        # `<=>` is pgvector's cosine *distance* (0 = same direction). Ordering by the raw
        # operator expression is what lets Postgres use the HNSW index.
        distance = Chunk.embedding.cosine_distance(query_vector)
        stmt = _select_chunks(1 - distance).order_by(distance).limit(k)
        async with self._sessions() as session:
            return _to_results((await session.execute(stmt)).all())


class KeywordRetriever:
    """Lexical retrieval with Postgres full-text search (exact terms, identifiers, names).

    The question is normalised with the same `english` configuration as the index (lower-case,
    stop words removed, stemmed), then the remaining terms are OR-ed: a natural-language
    question rarely has *every* word in one passage, so AND (`websearch_to_tsquery`) would
    return almost nothing. `ts_rank_cd` then favours chunks matching more terms, closer
    together, and in the heading (weight A).

    Postgres ranking has no IDF (unlike BM25): a rare identifier counts no more than a common
    word. Code identifiers in the question (`hnsw.ef_search`, `vector_l2_ops`) are therefore
    also matched as exact phrases, and a phrase hit outranks any bag-of-words score.
    """

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]):
        self._sessions = session_factory

    async def search(self, query: str, k: int) -> list[RetrievedChunk]:
        terms = select(func.unnest(func.to_tsvector("english", query)).column_valued("lexeme"))
        # Lexemes are already stemmed, so build the tsquery with the no-op `simple` config.
        or_query = func.to_tsquery(
            "simple",
            select(func.string_agg(func.quote_literal(literal_column("lexeme")), " | "))
            .select_from(terms.subquery())
            .scalar_subquery(),
        )
        search_vector = Chunk.search_vector
        rank: ColumnElement[float] = func.ts_rank_cd(search_vector, or_query, type_=Float)
        if identifiers := _IDENTIFIER.findall(query):
            phrases = [func.phraseto_tsquery("english", ident) for ident in identifiers]
            phrase_query: ColumnElement[Any] = phrases[0]
            for phrase in phrases[1:]:
                phrase_query = phrase_query.op("||")(phrase)
            rank = rank + func.coalesce(cast(search_vector.bool_op("@@")(phrase_query), Integer), 0)
        stmt = (
            _select_chunks(rank)
            .where(search_vector.bool_op("@@")(or_query))
            .order_by(rank.desc())
            .limit(k)
        )
        async with self._sessions() as session:
            return _to_results((await session.execute(stmt)).all())


class HybridRetriever:
    """Fuse dense and keyword results with Reciprocal Rank Fusion (RRF).

    RRF scores each chunk as sum(1 / (rrf_k + rank)) over the lists it appears in. It only uses
    *ranks*, which sidesteps the fact that cosine similarity and ts_rank live on unrelated
    scales and cannot be added directly. rrf_k = 60 is the value from the original paper
    (Cormack et al., 2009); it damps the advantage of being #1 in a single list.
    """

    def __init__(
        self, retrievers: Sequence[Retriever], candidates: int = 30, rrf_k: int = 60
    ) -> None:
        self._retrievers = retrievers
        self._candidates = candidates
        self._rrf_k = rrf_k

    async def search(self, query: str, k: int) -> list[RetrievedChunk]:
        result_lists = await asyncio.gather(
            *(r.search(query, self._candidates) for r in self._retrievers)
        )
        fused: dict[uuid.UUID, float] = {}
        chunks: dict[uuid.UUID, RetrievedChunk] = {}
        for results in result_lists:
            for rank, chunk in enumerate(results, start=1):
                fused[chunk.chunk_id] = fused.get(chunk.chunk_id, 0.0) + 1 / (self._rrf_k + rank)
                chunks.setdefault(chunk.chunk_id, chunk)
        best = sorted(fused, key=fused.__getitem__, reverse=True)[:k]
        return [replace(chunks[cid], score=fused[cid]) for cid in best]
