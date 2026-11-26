import json

import httpx
import pytest

from agentic_rag.embeddings import EmbeddingError, OllamaEmbedder


def _embedder(handler: httpx.MockTransport, dim: int = 3, batch_size: int = 2) -> OllamaEmbedder:
    client = httpx.AsyncClient(transport=handler)
    return OllamaEmbedder(client, "http://ollama", "bge-m3", dim=dim, batch_size=batch_size)


async def test_embed_batches_requests_and_preserves_order() -> None:
    batches: list[list[str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        inputs = json.loads(request.content)["input"]
        batches.append(inputs)
        return httpx.Response(200, json={"embeddings": [[float(len(t)), 0, 0] for t in inputs]})

    embedder = _embedder(httpx.MockTransport(handler))

    vectors = await embedder.embed(["a", "bb", "ccc"])

    assert batches == [["a", "bb"], ["ccc"]]
    assert [v[0] for v in vectors] == [1.0, 2.0, 3.0]


async def test_wrong_dimension_raises() -> None:
    transport = httpx.MockTransport(lambda _: httpx.Response(200, json={"embeddings": [[1.0]]}))

    with pytest.raises(EmbeddingError, match="wrong size"):
        await _embedder(transport).embed(["a"])


async def test_http_error_is_wrapped() -> None:
    transport = httpx.MockTransport(lambda _: httpx.Response(500))

    with pytest.raises(EmbeddingError, match="request failed"):
        await _embedder(transport).embed(["a"])
