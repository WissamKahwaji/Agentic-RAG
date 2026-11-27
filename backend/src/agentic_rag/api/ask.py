"""Question answering endpoint."""

from typing import Annotated, cast

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from agentic_rag.answering import Answer, HistoryTurn, Knowledge, QuestionAnswerer
from agentic_rag.config import Settings, get_settings
from agentic_rag.embeddings import EmbeddingError
from agentic_rag.llm import LLMError

router = APIRouter(tags=["qa"])


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    history: list[HistoryTurn] = Field(
        default_factory=list,
        max_length=5,
        description="Earlier turns, oldest first, so follow-up questions can be understood.",
    )
    knowledge: Knowledge | None = Field(
        default=None,
        description='"documents" (only your documents) or "hybrid" (documents first, then web '
        "or model knowledge, always labelled). Defaults to the server's KNOWLEDGE_MODE.",
    )


def get_answer_service(request: Request) -> QuestionAnswerer:
    return cast(QuestionAnswerer, request.app.state.answer_service)


@router.post("/ask", response_model=Answer)
async def ask(
    body: AskRequest,
    service: Annotated[QuestionAnswerer, Depends(get_answer_service)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> Answer:
    knowledge = body.knowledge or settings.knowledge_mode
    if knowledge == "hybrid" and settings.answer_mode != "agentic":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Hybrid knowledge needs the agentic answer mode (ANSWER_MODE=agentic).",
        )
    try:
        return await service.answer(
            body.question.strip(), top_k=body.top_k, history=body.history, knowledge=knowledge
        )
    except (LLMError, EmbeddingError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The model server is unavailable. Check that Ollama is running.",
        ) from exc
