"""Deterministic test doubles for the model and retrieval protocols."""

import uuid

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
    model = "fake-llm"

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.received: list[list[ChatMessage]] = []

    async def chat(self, messages: list[ChatMessage], *, temperature: float = 0.0) -> LLMResult:
        self.received.append(messages)
        return LLMResult(text=self.reply, model=self.model)
