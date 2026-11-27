"""Text embedding behind a small protocol, so the vendor can change without touching callers."""

import asyncio
import threading
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


class SentenceTransformerEmbedder:
    """The same embedding model run in-process (no Ollama), e.g. on a CPU-only server.

    For bge-m3 the vectors match Ollama's (cosine 1.000 on spot checks), so an index built with
    one can be queried with the other without re-embedding.
    """

    def __init__(
        self, model_name: str, dim: int, device: str | None = None, name: str | None = None
    ) -> None:
        # `model` names the model for ingestion fingerprints; pass the Ollama name ("bge-m3")
        # when vectors are interchangeable, so switching providers does not force a re-index.
        self.model = name or model_name
        self._hf_model = model_name
        self.dim = dim
        self._device = device
        self._model: object | None = None
        self._lock = threading.Lock()

    def _encode_sync(self, texts: list[str]) -> list[list[float]]:
        with self._lock:
            if self._model is None:
                from sentence_transformers import SentenceTransformer  # heavy, load lazily

                self._model = SentenceTransformer(self._hf_model, device=self._device)
            vectors = self._model.encode(  # type: ignore[attr-defined]
                texts, normalize_embeddings=True, show_progress_bar=False
            )
        result = [[float(x) for x in v] for v in vectors]
        if any(len(v) != self.dim for v in result):
            raise EmbeddingError(f"model {self.model!r} returned vectors of the wrong size")
        return result

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        return await asyncio.to_thread(self._encode_sync, list(texts))
