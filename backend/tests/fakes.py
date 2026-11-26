"""Deterministic test doubles for the model and retrieval protocols."""

import uuid
from collections.abc import Callable
from typing import Any

from agentic_rag.llm import ChatMessage, LLMResult
from agentic_rag.retrieval import RetrievedChunk


def make_chunk(
    content: str = "pgvector supports HNSW indexes.", **overrides: object
) -> RetrievedChunk:
    fields: dict[str, object] = {
        "chunk_id": uuid.uuid4(),
        "source": "pgvector/README.md",
        "title": "pgvector",
        "page": None,
        "section": "pgvector > HNSW",
        "content": content,
        "score": 0.8,
    }
    fields.update(overrides)
    return RetrievedChunk(**fields)  # type: ignore[arg-type]


class FakeRetriever:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = chunks
        self.calls: list[tuple[str, int]] = []

    async def search(self, query: str, k: int) -> list[RetrievedChunk]:
        self.calls.append((query, k))
        return self.chunks[:k]


class FakeLLM:
    """Replies with a fixed string, or with `reply(messages, json_schema)` when it is callable."""

    model = "fake-llm"

    def __init__(
        self, reply: str | Callable[[list[ChatMessage], dict[str, Any] | None], str]
    ) -> None:
        self.reply = reply
        self.received: list[list[ChatMessage]] = []
        self.schemas: list[dict[str, Any] | None] = []

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.0,
        json_schema: dict[str, Any] | None = None,
    ) -> LLMResult:
        self.received.append(messages)
        self.schemas.append(json_schema)
        text = self.reply(messages, json_schema) if callable(self.reply) else self.reply
        return LLMResult(text=text, model=self.model)
