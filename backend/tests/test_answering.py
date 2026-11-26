from agentic_rag.answering import AnswerService, is_refusal, resolve_citations
from agentic_rag.prompts import CITATION_REMINDER, NO_ANSWER, grounded_answer_user_message
from tests.fakes import FakeLLM, FakeRetriever, make_chunk


def test_resolve_citations_maps_markers_in_order_of_first_use() -> None:
    chunks = [make_chunk("a"), make_chunk("b"), make_chunk("c")]

    text, citations = resolve_citations("C first [3]. Then A [1][3].", chunks)

    assert text == "C first [3]. Then A [1][3]."
    assert [c.number for c in citations] == [3, 1]
    assert citations[0].content == "c"


def test_resolve_citations_strips_markers_with_no_source() -> None:
    chunks = [make_chunk("a")]

    text, citations = resolve_citations("Real [1]. Invented [7].", chunks)

    assert text == "Real [1]. Invented."
    assert [c.number for c in citations] == [1]


def test_resolve_citations_handles_grouped_markers() -> None:
    chunks = [make_chunk("a"), make_chunk("b")]

    text, citations = resolve_citations("Both say so [1, 2].", chunks)

    assert text == "Both say so [1][2]."
    assert [c.number for c in citations] == [1, 2]


def test_is_refusal_tolerates_case_and_trailing_punctuation() -> None:
    assert is_refusal("i don't know based on the provided documents")
    assert not is_refusal("HNSW is a graph index [1].")


def test_prompt_numbers_sources_with_location() -> None:
    chunks = [make_chunk("first"), make_chunk("second", page=6, section=None, title="RAG paper")]

    message = grounded_answer_user_message("What?", chunks)

    assert "[1] pgvector (pgvector > HNSW)\nfirst" in message
    assert "[2] RAG paper (page 6)\nsecond" in message
    assert "Question: What?" in message
    assert message.endswith(CITATION_REMINDER)


async def test_answer_returns_grounded_answer_with_citations() -> None:
    retriever = FakeRetriever([make_chunk("HNSW is a graph index.")])
    llm = FakeLLM("HNSW is a graph-based index [1].")
    service = AnswerService(retriever, llm, top_k=5, temperature=0.0)

    answer = await service.answer("What is HNSW?")

    assert answer.grounded
    assert [c.number for c in answer.citations] == [1]
    assert answer.prompt == "grounded_answer@v2"
    assert retriever.calls == [("What is HNSW?", 5)]
    assert "HNSW is a graph index." in llm.received[0][1].content


async def test_answer_with_no_citations_is_not_grounded() -> None:
    service = AnswerService(
        FakeRetriever([make_chunk()]), FakeLLM("It is fast."), top_k=5, temperature=0.0
    )

    answer = await service.answer("Is it fast?")

    assert not answer.grounded


async def test_refusal_is_not_grounded_and_has_no_citations() -> None:
    service = AnswerService(
        FakeRetriever([make_chunk()]), FakeLLM(f"{NO_ANSWER} [1]"), top_k=5, temperature=0.0
    )

    answer = await service.answer("Capital of France?")

    assert not answer.grounded
    assert answer.citations == []
    assert answer.answer == NO_ANSWER


async def test_empty_index_refuses_without_calling_the_model() -> None:
    llm = FakeLLM("should not be used")
    service = AnswerService(FakeRetriever([]), llm, top_k=5, temperature=0.0)

    answer = await service.answer("Anything?")

    assert answer.answer == NO_ANSWER
    assert llm.received == []


async def test_top_k_override_is_passed_to_retriever() -> None:
    retriever = FakeRetriever([make_chunk()])
    service = AnswerService(retriever, FakeLLM("x [1]"), top_k=5, temperature=0.0)

    await service.answer("q?", top_k=2)

    assert retriever.calls == [("q?", 2)]
