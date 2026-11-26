"""Read-only view of the ingested corpus."""

import uuid
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from agentic_rag.api.deps import get_session
from agentic_rag.models import Chunk, Document

router = APIRouter(prefix="/documents", tags=["documents"])


class DocumentSummary(BaseModel):
    id: uuid.UUID
    source: str
    title: str
    media_type: str
    chunk_count: int
    updated_at: datetime


@router.get("", response_model=list[DocumentSummary])
async def list_documents(
    session: Annotated[AsyncSession, Depends(get_session)],
) -> list[DocumentSummary]:
    stmt = (
        select(Document, func.count(Chunk.id))
        .outerjoin(Chunk, Chunk.document_id == Document.id)
        .group_by(Document.id)
        .order_by(Document.source)
    )
    rows = (await session.execute(stmt)).all()
    return [
        DocumentSummary(
            id=doc.id,
            source=doc.source,
            title=doc.title,
            media_type=doc.media_type,
            chunk_count=count,
            updated_at=doc.updated_at,
        )
        for doc, count in rows
    ]
