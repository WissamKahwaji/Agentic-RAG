"""Integration: runs the pipeline against the real Postgres, with a fake embedder."""

import uuid
from collections.abc import AsyncIterator, Sequence
from pathlib import Path

import pytest
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from agentic_rag import db
from agentic_rag.config import get_settings
from agentic_rag.ingestion import IngestionService
from agentic_rag.models import EMBEDDING_DIM, Chunk, Document
from agentic_rag.retrieval import KeywordRetriever

pytestmark = pytest.mark.integration


class FakeEmbedder:
    model = "fake"
    dim = EMBEDDING_DIM

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[0.1] * self.dim for _ in texts]


@pytest.fixture
async def sessions() -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    engine = db.create_engine(str(get_settings().database_url))
    yield db.create_session_factory(engine)
    await engine.dispose()


@pytest.fixture
async def corpus(tmp_path: Path, sessions: async_sessionmaker[AsyncSession]) -> AsyncIterator[Path]:
    root = tmp_path / f"test-{uuid.uuid4().hex[:8]}"
    root.mkdir()
    yield root
    async with sessions() as s, s.begin():
        await s.execute(delete(Document).where(Document.source.like(f"{root.name}/%")))


async def _chunk_count(sessions: async_sessionmaker[AsyncSession], source: str) -> int:
    async with sessions() as s:
        stmt = select(func.count(Chunk.id)).join(Document).where(Document.source == source)
        return (await s.scalar(stmt)) or 0


async def test_ingest_is_idempotent_and_replaces_changed_documents(
    corpus: Path, sessions: async_sessionmaker[AsyncSession]
) -> None:
    service = IngestionService(sessions, FakeEmbedder(), chunk_size=200, chunk_overlap=0)
    path = corpus / "notes.md"
    source = f"{corpus.name}/notes.md"
    path.write_text("# Notes\n\n" + "first version of the text. " * 30)

    first = await service.ingest_file(path, source=source)
    again = await service.ingest_file(path, source=source)
    path.write_text("# Notes\n\nshort second version, long enough to keep.")
    changed = await service.ingest_file(path, source=source)

    assert first.status == "created"
    assert first.chunks > 1
    assert again.status == "unchanged"
    assert changed.status == "updated"
    assert await _chunk_count(sessions, source) == changed.chunks == 1


async def test_keyword_search_finds_exact_identifier(
    corpus: Path, sessions: async_sessionmaker[AsyncSession]
) -> None:
    service = IngestionService(sessions, FakeEmbedder(), chunk_size=400, chunk_overlap=0)
    path = corpus / "settings.md"
    path.write_text(
        "# Settings\n\n## Tuning\n\nRaise zqxflux_window_size to keep more history in memory.\n\n"
        "## Other\n\nThis section talks about something unrelated entirely, nothing to see."
    )
    await service.ingest_file(path, source=f"{corpus.name}/settings.md")

    results = await KeywordRetriever(sessions).search("what does zqxflux_window_size do?", k=3)

    assert results, "expected a keyword match"
    assert results[0].section == "Settings > Tuning"
