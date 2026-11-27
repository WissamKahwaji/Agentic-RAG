"""Second-stage reranking with a cross-encoder.

Retrievers embed the question and each passage *separately* (bi-encoder), which is fast enough
to search millions of chunks but compares two compressed summaries. A cross-encoder reads the
question and one passage *together* in a single forward pass, so every question word can attend
to every passage word. That is far more accurate and far too slow for a whole corpus, so it only
re-orders a short candidate list produced by cheaper retrievers.
"""

import asyncio
import logging
import threading
from dataclasses import replace
from typing import Protocol

from agentic_rag.retrieval import RetrievedChunk, Retriever

logger = logging.getLogger(__name__)


class Reranker(Protocol):
    async def score(self, query: str, passages: list[str]) -> list[float]:
        """Relevance of each passage to the query, higher is better, in input order."""
        ...


class CrossEncoderReranker:
    """Local cross-encoder via sentence-transformers (e.g. BAAI/bge-reranker-v2-m3).

    The model is loaded lazily on first use and runs on Apple Silicon's GPU (MPS) when
    available. Inference is synchronous PyTorch, so it runs in a worker thread to keep the
    event loop responsive; a lock serialises calls because the model is not thread-safe.
    """

    def __init__(
        self,
        model_name: str,
        max_length: int = 512,
        batch_size: int = 16,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self._device = device
        self._max_length = max_length
        self._batch_size = batch_size
        self._model: object | None = None
        self._lock = threading.Lock()

    def _load(self) -> object:
        if self._model is None:
            import torch  # heavy imports deferred so the API starts fast without reranking
            from sentence_transformers import CrossEncoder

            device = self._device or ("mps" if torch.backends.mps.is_available() else "cpu")
            logger.info("loading reranker %s on %s", self.model_name, device)
            # Some cross-encoders output probabilities (bge rerankers), others raw logits
            # (ms-marco MiniLM). A sigmoid on every model puts all scores on one 0-1 scale, so a
            # relevance threshold means the same thing whichever model is configured.
            self._model = CrossEncoder(
                self.model_name,
                max_length=self._max_length,
                device=device,
                activation_fn=torch.nn.Sigmoid(),
            )
        return self._model

    def _score_sync(self, query: str, passages: list[str]) -> list[float]:
        with self._lock:
            model = self._load()
            scores = model.predict(  # type: ignore[attr-defined]
                [(query, p) for p in passages],
                batch_size=self._batch_size,
                show_progress_bar=False,
            )
        return [float(s) for s in scores]

    async def warm_up(self) -> None:
        await self.score("warm-up", ["warm-up"])

    async def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        return await asyncio.to_thread(self._score_sync, query, passages)


class RerankingRetriever:
    """Fetch `candidates` from a first-stage retriever, keep the top `k` by reranker score."""

    def __init__(self, base: Retriever, reranker: Reranker, candidates: int = 20) -> None:
        self._base = base
        self._reranker = reranker
        self._candidates = candidates

    async def search(self, query: str, k: int) -> list[RetrievedChunk]:
        candidates = await self._base.search(query, max(k, self._candidates))
        scores = await self._reranker.score(query, [_passage_text(c) for c in candidates])
        ranked = sorted(zip(candidates, scores, strict=True), key=lambda p: p[1], reverse=True)
        return [replace(chunk, score=score) for chunk, score in ranked[:k]]


def _passage_text(chunk: RetrievedChunk) -> str:
    # Same idea as contextual embeddings: the heading path tells the reranker what the
    # passage is about when its text alone is ambiguous.
    return f"{chunk.title} > {chunk.location}\n{chunk.content}"
