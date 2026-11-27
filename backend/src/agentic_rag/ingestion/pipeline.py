"""Ingestion pipeline: load -> chunk -> embed -> store, idempotent per source."""

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from agentic_rag.embeddings import Embedder
from agentic_rag.ingestion.chunking import chunk_document
from agentic_rag.ingestion.loaders import LoadedDocument, is_supported, load_file
from agentic_rag.models import Chunk, Document

logger = logging.getLogger(__name__)

IngestStatus = Literal["created", "updated", "unchanged", "skipped"]

# Bump when loading or chunking logic changes, so unchanged files are still re-processed.
PIPELINE_VERSION = 4


@dataclass(frozen=True)
class IngestResult:
    source: str
    status: IngestStatus
    chunks: int = 0
    seconds: float = 0.0
    detail: str | None = None


class IngestionService:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        embedder: Embedder,
        chunk_size: int,
        chunk_overlap: int,
    ) -> None:
        self._sessions = session_factory
        self._embedder = embedder
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def fingerprint(self, content_hash: str) -> str:
        """Identify *how* a file was indexed, not just its bytes.

        Skipping re-ingestion on an identical file hash alone is a classic bug: after changing
        the chunker, the embedding model or chunk size, old vectors silently stay in the index.
        """
        recipe = (
            f"{content_hash}|v{PIPELINE_VERSION}|{self._embedder.model}|{self._embedder.dim}"
            f"|{self._chunk_size}|{self._chunk_overlap}"
        )
        return hashlib.sha256(recipe.encode()).hexdigest()

    async def ingest_path(self, root: Path) -> list[IngestResult]:
        """Ingest a single file, or every supported file under a directory (recursively)."""
        if await asyncio.to_thread(root.is_file):
            return [await self.ingest_file(root, source=root.name)]
        files = await asyncio.to_thread(_discover, root)
        return [await self.ingest_file(p, source=p.relative_to(root).as_posix()) for p in files]

    async def ingest_file(self, path: Path, source: str | None = None) -> IngestResult:
        started = time.perf_counter()
        doc = await asyncio.to_thread(load_file, path, source)

        if doc.is_empty:
            logger.warning("no extractable text in %s (scanned PDF?)", doc.source)
            return IngestResult(doc.source, "skipped", detail="no extractable text")

        async with self._sessions() as session:
            existing_hash = await session.scalar(
                select(Document.content_hash).where(Document.source == doc.source)
            )
        fingerprint = self.fingerprint(doc.content_hash)
        if existing_hash == fingerprint:
            return IngestResult(doc.source, "unchanged", seconds=time.perf_counter() - started)

        drafts = chunk_document(doc, self._chunk_size, self._chunk_overlap)
        # Embed before opening the write transaction: it is the slow, failure-prone step and
        # must not hold database locks while it runs.
        vectors = await self._embedder.embed([d.embedding_text for d in drafts])
        chunks = [
            Chunk(
                chunk_index=d.index,
                content=d.content,
                page=d.page,
                section=d.section,
                embedding=vector,
            )
            for d, vector in zip(drafts, vectors, strict=True)
        ]

        status = await self._replace(doc, fingerprint, chunks)
        elapsed = time.perf_counter() - started
        logger.info("%s %s: %d chunks in %.1fs", status, doc.source, len(chunks), elapsed)
        return IngestResult(doc.source, status, chunks=len(chunks), seconds=elapsed)

    async def _replace(
        self, doc: LoadedDocument, fingerprint: str, chunks: list[Chunk]
    ) -> IngestStatus:
        """Insert the document, or swap all chunks of an existing one, in one transaction."""
        async with self._sessions() as session, session.begin():
            document = await session.scalar(
                select(Document).where(Document.source == doc.source).with_for_update()
            )
            if document is None:
                document = Document(source=doc.source)
                session.add(document)
                status: IngestStatus = "created"
            else:
                await session.execute(delete(Chunk).where(Chunk.document_id == document.id))
                status = "updated"

            document.title = doc.title
            document.media_type = doc.media_type
            document.content_hash = fingerprint
            await session.flush()

            for chunk in chunks:
                chunk.document_id = document.id
            session.add_all(chunks)
        return status


def _discover(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and is_supported(p))
