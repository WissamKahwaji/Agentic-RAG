"""Dense (vector) retrieval over stored chunks."""

import logging
import uuid
from dataclasses import dataclass
from typing import Protocol

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from agentic_rag.embeddings import Embedder
from agentic_rag.models import Chunk, Document

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: uuid.UUID
    source: str
    title: str
    page: int | None
    section: str | None
    content: str
    score: float  # cosine similarity, higher is closer (1.0 = identical direction)

    @property
    def location(self) -> str:
        """Human-readable citation location, e.g. `page 6` or `pgvector > HNSW`."""
        if self.page is not None:
            return f"page {self.page}"
        return self.section or self.title


class Retriever(Protocol):
    async def search(self, query: str, k: int) -> list[RetrievedChunk]: ...


class VectorRetriever:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession], embedder: Embedder):
        self._sessions = session_factory
        self._embedder = embedder

    async def search(self, query: str, k: int) -> list[RetrievedChunk]:
        [query_vector] = await self._embedder.embed([query])
        # `<=>` is pgvector's cosine *distance* (0 = same direction). Ordering by the raw
        # operator expression is what lets Postgres use the HNSW index.
        distance = Chunk.embedding.cosine_distance(query_vector)
        stmt = (
            select(Chunk, Document.source, Document.title, distance.label("distance"))
            .join(Document, Document.id == Chunk.document_id)
            .order_by(distance)
            .limit(k)
        )
        async with self._sessions() as session:
            rows = (await session.execute(stmt)).all()

        results = [
            RetrievedChunk(
                chunk_id=chunk.id,
                source=source,
                title=title,
                page=chunk.page,
                section=chunk.section,
                content=chunk.content,
                score=1.0 - float(dist),
            )
            for chunk, source, title, dist in rows
        ]
        logger.debug(
            "retrieved %d chunks for %r: %s",
            len(results),
            query,
            [(r.source, round(r.score, 3)) for r in results],
        )
        return results
