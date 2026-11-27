"""CLI: `uv run python -m agentic_rag.ingestion <file-or-directory>`."""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

import httpx

from agentic_rag import db
from agentic_rag.config import get_settings
from agentic_rag.ingestion.pipeline import IngestionService, IngestResult
from agentic_rag.providers import build_embedder


async def run(path: Path) -> list[IngestResult]:
    settings = get_settings()
    engine = db.create_engine(str(settings.database_url))
    try:
        async with httpx.AsyncClient(timeout=settings.http_timeout_s) as http:
            service = IngestionService(
                session_factory=db.create_session_factory(engine),
                embedder=build_embedder(settings, http),
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
            )
            return await service.ingest_path(path)
    finally:
        await engine.dispose()


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest documents into the vector store.")
    parser.add_argument("path", type=Path, help="file or directory (searched recursively)")
    args = parser.parse_args()

    logging.basicConfig(level=get_settings().log_level, format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    if not args.path.exists():
        parser.error(f"path not found: {args.path}")

    results = asyncio.run(run(args.path))

    width = max((len(r.source) for r in results), default=10)
    for r in results:
        note = f"  ({r.detail})" if r.detail else ""
        print(f"{r.source:<{width}}  {r.status:<9}  {r.chunks:>4} chunks  {r.seconds:5.1f}s{note}")
    total = sum(r.chunks for r in results if r.status in ("created", "updated"))
    print(f"\n{len(results)} files, {total} chunks written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
