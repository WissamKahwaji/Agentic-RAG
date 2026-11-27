from agentic_rag.evals.metrics import GoldenQuestion, RelevantSpan, first_relevant_rank, summarize
from tests.fakes import make_chunk


def test_section_label_matches_itself_and_subsections_only() -> None:
    span = RelevantSpan(source="pgvector/README.md", section="pgvector > HNSW")

    assert span.matches(make_chunk(section="pgvector > HNSW"))
    assert span.matches(make_chunk(section="pgvector > HNSW > Index Options"))
    assert not span.matches(make_chunk(section="pgvector > HNSWX"))
    assert not span.matches(make_chunk(section="pgvector > IVFFlat"))
    assert not span.matches(make_chunk(source="other.md", section="pgvector > HNSW"))


def test_page_label_matches_listed_pages() -> None:
    span = RelevantSpan(source="paper.pdf", pages=[3, 4])

    assert span.matches(make_chunk(source="paper.pdf", page=4, section=None))
    assert not span.matches(make_chunk(source="paper.pdf", page=5, section=None))


def test_first_relevant_rank_and_summary() -> None:
    q = GoldenQuestion(
        id="q",
        kind="natural",
        question="?",
        relevant=[RelevantSpan(source="pgvector/README.md", section="pgvector > HNSW")],
    )
    results = [make_chunk(section="pgvector > IVFFlat"), make_chunk(section="pgvector > HNSW")]

    assert first_relevant_rank(q, results) == 2
    assert first_relevant_rank(q, results[:1]) is None

    s = summarize({"a": 1, "b": 2, "c": None, "d": 7}, [10.0, 30.0])
    assert (s.hit_at_1, s.hit_at_5) == (0.25, 0.5)
    assert s.mrr_at_10 == (1 + 0.5 + 0 + 1 / 7) / 4
    assert s.p50_ms == 20.0
    assert s.misses == ["c", "d"]


def test_contains_requires_the_answer_text_in_the_passage() -> None:
    span = RelevantSpan(source="paper.pdf", pages=[3], contains=["BART-large"])

    assert span.matches(make_chunk("We use BART-large.", source="paper.pdf", page=3, section=None))
    assert not span.matches(make_chunk("Other text.", source="paper.pdf", page=3, section=None))
