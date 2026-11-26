"""Question answering endpoint."""

from typing import Annotated, cast

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from agentic_rag.answering import Answer, HistoryTurn, QuestionAnswerer
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


def get_answer_service(request: Request) -> QuestionAnswerer:
    return cast(QuestionAnswerer, request.app.state.answer_service)


@router.post("/ask", response_model=Answer)
async def ask(
    body: AskRequest,
    service: Annotated[QuestionAnswerer, Depends(get_answer_service)],
) -> Answer:
    try:
        return await service.answer(body.question.strip(), top_k=body.top_k, history=body.history)
    except (LLMError, EmbeddingError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The model server is unavailable. Check that Ollama is running.",
        ) from exc
