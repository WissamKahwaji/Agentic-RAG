"""Text embedding behind a small protocol, so the vendor can change without touching callers."""

from collections.abc import Sequence
from typing import Protocol

import httpx


class EmbeddingError(RuntimeError):
    pass


class Embedder(Protocol):
    model: str
    dim: int

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one vector of length `dim` per input text, in input order."""
        ...


class OllamaEmbedder:
    """Embeds through Ollama's `/api/embed`, which accepts a batch of inputs per request."""

    def __init__(
        self,
        http: httpx.AsyncClient,
        base_url: str,
        model: str,
        dim: int,
        batch_size: int = 32,
    ) -> None:
        self._http = http
        self._url = f"{base_url.rstrip('/')}/api/embed"
        self.model = model
        self.dim = dim
        self._batch_size = batch_size

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = list(texts[start : start + self._batch_size])
            vectors.extend(await self._embed_batch(batch))
        return vectors

    async def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        try:
            resp = await self._http.post(self._url, json={"model": self.model, "input": batch})
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise EmbeddingError(f"Ollama embedding request failed: {exc}") from exc

        vectors: list[list[float]] = resp.json().get("embeddings", [])
        if len(vectors) != len(batch):
            raise EmbeddingError(f"expected {len(batch)} embeddings, got {len(vectors)}")
        if any(len(v) != self.dim for v in vectors):
            raise EmbeddingError(
                f"model {self.model!r} returned vectors of the wrong size; expected {self.dim}"
            )
        return vectors
