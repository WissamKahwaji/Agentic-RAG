import httpx
import pytest
from fastapi import FastAPI

from agentic_rag.answering import AnswerService
from agentic_rag.api.ask import get_answer_service
from agentic_rag.llm import ChatMessage, LLMError, LLMResult
from tests.fakes import FakeLLM, FakeRetriever, make_chunk


class FailingLLM:
    model = "down"

    async def chat(self, messages: list[ChatMessage], *, temperature: float = 0.0) -> LLMResult:
        raise LLMError("connection refused")


def _use(app: FastAPI, service: AnswerService) -> None:
    app.dependency_overrides[get_answer_service] = lambda: service


async def test_ask_returns_answer_and_citations(app: FastAPI, client: httpx.AsyncClient) -> None:
    _use(app, AnswerService(FakeRetriever([make_chunk()]), FakeLLM("Yes [1]."), 5, 0.0))

    resp = await client.post("/ask", json={"question": "Does pgvector support HNSW?"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["grounded"] is True
    assert body["citations"][0]["source"] == "pgvector/README.md"
    assert set(body["timings"]) == {"retrieval_ms", "generation_ms"}


async def test_ask_rejects_too_short_question(app: FastAPI, client: httpx.AsyncClient) -> None:
    resp = await client.post("/ask", json={"question": "?"})

    assert resp.status_code == 422


async def test_ask_returns_503_when_model_is_down(app: FastAPI, client: httpx.AsyncClient) -> None:
    _use(app, AnswerService(FakeRetriever([make_chunk()]), FailingLLM(), 5, 0.0))

    resp = await client.post("/ask", json={"question": "Does pgvector support HNSW?"})

    assert resp.status_code == 503


@pytest.mark.integration
async def test_ask_end_to_end_against_ingested_corpus(client: httpx.AsyncClient) -> None:
    """Structure only: real model output varies, so wording is checked by evals, not here."""
    resp = await client.post("/ask", json={"question": "How do I create an HNSW index?"})

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["grounded"] is True, body["answer"]
    assert any(c["source"].startswith("pgvector/") for c in body["citations"])
