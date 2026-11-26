import pytest

from agentic_rag.config import get_settings
from agentic_rag.reranking import RerankingRetriever
from agentic_rag.retrieval import HybridRetriever, KeywordRetriever, RetrievedChunk, VectorRetriever
from agentic_rag.search import RETRIEVAL_MODES, build_retriever
from tests.fakes import FakeRetriever, make_chunk


class FakeReranker:
    """Scores a passage by how many times it contains the word 'good'."""

    def __init__(self) -> None:
        self.seen: list[int] = []

    async def score(self, query: str, passages: list[str]) -> list[float]:
        self.seen.append(len(passages))
        return [float(p.count("good")) for p in passages]


def _ids(results: list[RetrievedChunk]) -> list[str]:
    return [r.content for r in results]


async def test_rrf_rewards_chunks_found_by_both_retrievers() -> None:
    a, b, c, d = (make_chunk(x) for x in "abcd")
    dense = FakeRetriever([a, b, c])
    keyword = FakeRetriever([d, c, a])

    results = await HybridRetriever([dense, keyword], candidates=3).search("q", k=4)

    # a: 1/61 + 1/63, c: 1/63 + 1/62 beat b (1/62) and d (1/61), which appear once.
    assert _ids(results)[:2] == ["a", "c"]
    assert set(_ids(results)) == {"a", "b", "c", "d"}


async def test_rrf_requests_candidates_from_each_retriever_and_truncates_to_k() -> None:
    dense = FakeRetriever([make_chunk(str(i)) for i in range(10)])
    keyword = FakeRetriever([])

    results = await HybridRetriever([dense, keyword], candidates=7).search("q", k=3)

    assert dense.calls == [("q", 7)]
    assert keyword.calls == [("q", 7)]
    assert len(results) == 3


async def test_reranker_reorders_candidates_and_replaces_scores() -> None:
    base = FakeRetriever([make_chunk("meh"), make_chunk("good good"), make_chunk("good")])
    reranker = FakeReranker()

    results = await RerankingRetriever(base, reranker, candidates=3).search("q", k=2)

    assert _ids(results) == ["good good", "good"]
    assert [r.score for r in results] == [2.0, 1.0]
    assert reranker.seen == [3]


async def test_reranker_fetches_at_least_k_candidates() -> None:
    base = FakeRetriever([make_chunk(str(i)) for i in range(10)])

    await RerankingRetriever(base, FakeReranker(), candidates=2).search("q", k=5)

    assert base.calls == [("q", 5)]


def test_build_retriever_composes_stages() -> None:
    settings = get_settings()
    args = (None, None, settings)  # session factory / embedder are unused until search()

    assert isinstance(build_retriever("dense", *args), VectorRetriever)  # type: ignore[arg-type]
    assert isinstance(build_retriever("keyword", *args), KeywordRetriever)  # type: ignore[arg-type]
    assert isinstance(build_retriever("hybrid", *args), HybridRetriever)  # type: ignore[arg-type]
    reranked = build_retriever("hybrid+rerank", None, None, settings, FakeReranker())  # type: ignore[arg-type]
    assert isinstance(reranked, RerankingRetriever)
    assert "hybrid+rerank" in RETRIEVAL_MODES


def test_build_retriever_rejects_rerank_without_reranker() -> None:
    with pytest.raises(ValueError, match="needs a reranker"):
        build_retriever("dense+rerank", None, None, get_settings())  # type: ignore[arg-type]
