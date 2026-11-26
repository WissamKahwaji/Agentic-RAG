"""Routing tests for the agent graph, with scripted fakes for the model and the reranker."""

import json
from collections.abc import Iterator
from typing import Any

from agentic_rag.agent import AgentAnswerService, AgentConfig
from agentic_rag.answering import HistoryTurn
from agentic_rag.llm import ChatMessage
from agentic_rag.prompts import (
    CONDENSE_QUESTION,
    GROUNDED_ANSWER,
    NO_ANSWER,
    REFORMULATE_QUERY,
    VERIFY_CLAIM,
)
from tests.fakes import FakeLLM, FakeRetriever, make_chunk


class QueryReranker:
    """Scores every passage high only when the query contains `good_query`."""

    def __init__(self, good_query: str = "") -> None:
        self.good_query = good_query
        self.queries: list[str] = []

    async def score(self, query: str, passages: list[str]) -> list[float]:
        self.queries.append(query)
        return [0.9 if self.good_query in query else 0.01 for _ in passages]


class ScriptedLLM(FakeLLM):
    """Answers by prompt: each prompt name maps to an iterator of replies."""

    def __init__(self, **scripts: list[str]) -> None:
        self.scripts: dict[str, Iterator[str]] = {k: iter(v) for k, v in scripts.items()}
        self.calls: list[str] = []
        super().__init__(self._reply)

    def _reply(self, messages: list[ChatMessage], _schema: dict[str, Any] | None) -> str:
        names = {
            CONDENSE_QUESTION.system: "condense",
            REFORMULATE_QUERY.system: "reformulate",
            VERIFY_CLAIM.system: "self_check",
            GROUNDED_ANSWER.system: "generate",
        }
        name = names[messages[0].content]
        self.calls.append(name)
        return next(self.scripts[name])


def _verdict(supported: bool, *_claims: str) -> str:
    """Per-claim verifier reply; the flagged claim text comes from the answer itself."""
    return json.dumps({"supported": supported})


def _service(llm: ScriptedLLM, reranker: QueryReranker, **config: Any) -> AgentAnswerService:
    retriever = FakeRetriever([make_chunk("HNSW builds a graph."), make_chunk("Other text.")])
    return AgentAnswerService(retriever, reranker, llm, top_k=5, config=AgentConfig(**config))


def _steps(answer: Any) -> list[str]:
    return [s.step for s in answer.trace]


async def test_happy_path_retrieves_grades_generates_and_checks() -> None:
    llm = ScriptedLLM(generate=["HNSW builds a graph [1]."], self_check=[_verdict(True)])

    answer = await _service(llm, QueryReranker()).answer("What is HNSW?")

    assert _steps(answer) == ["retrieve", "grade", "generate", "self_check"]
    assert answer.grounded
    assert answer.self_check is not None
    assert answer.self_check.passed
    assert answer.search_query is None  # searched the question as asked


async def test_no_relevant_passages_reformulates_then_refuses_without_generating() -> None:
    llm = ScriptedLLM(reformulate=['{"query": "still unrelated"}'])

    answer = await _service(llm, QueryReranker(good_query="never")).answer("Unknown topic?")

    assert _steps(answer) == ["retrieve", "grade", "reformulate", "retrieve", "grade", "refuse"]
    assert answer.refused
    assert answer.answer == NO_ANSWER
    assert "generate" not in llm.calls


async def test_reformulated_query_can_rescue_retrieval() -> None:
    llm = ScriptedLLM(
        reformulate=['{"query": "hnsw index options"}'],
        generate=["Use m and ef_construction [1]."],
        self_check=[_verdict(True)],
    )
    reranker = QueryReranker(good_query="hnsw")

    answer = await _service(llm, reranker).answer("How do I tune the graph thing?")

    assert answer.grounded
    assert reranker.queries == ["How do I tune the graph thing?", "hnsw index options"]


async def test_failed_self_check_regenerates_with_feedback_once() -> None:
    llm = ScriptedLLM(
        generate=["HNSW is fast and free [1].", "HNSW builds a graph [1]."],
        self_check=[_verdict(False, "HNSW is free"), _verdict(True)],
    )

    answer = await _service(llm, QueryReranker()).answer("What is HNSW?")

    assert _steps(answer) == [
        "retrieve",
        "grade",
        "generate",
        "self_check",
        "generate",
        "self_check",
    ]
    assert answer.answer == "HNSW builds a graph [1]."
    assert answer.grounded
    revision = llm.received[2][-1].content  # third call = second generate
    assert "HNSW is fast and free" in revision


async def test_regeneration_is_bounded_and_flags_the_answer() -> None:
    llm = ScriptedLLM(
        generate=["HNSW runs on the GPU by default [1].", "HNSW needs no memory at all [1]."],
        self_check=[_verdict(False, "A"), _verdict(False, "B")],
    )

    answer = await _service(llm, QueryReranker(), max_generation_attempts=2).answer("Q?")

    assert llm.calls.count("generate") == 2
    assert not answer.grounded
    assert answer.self_check is not None
    assert answer.self_check.unsupported_claims == ["HNSW needs no memory at all."]


async def test_follow_up_is_condensed_using_history() -> None:
    llm = ScriptedLLM(
        condense=['{"query": "How do I create an IVFFlat index?"}'],
        generate=["Use lists [1]."],
        self_check=[_verdict(True)],
    )
    reranker = QueryReranker()
    history = [HistoryTurn(question="How do I create an HNSW index?", answer="Use hnsw [1].")]

    answer = await _service(llm, reranker).answer("What about IVFFlat?", history=history)

    assert _steps(answer)[0] == "condense"
    assert answer.search_query == "How do I create an IVFFlat index?"
    assert reranker.queries == ["How do I create an IVFFlat index?"]
    assert "How do I create an HNSW index?" in llm.received[0][1].content


async def test_self_check_can_be_disabled() -> None:
    llm = ScriptedLLM(generate=["HNSW builds a graph [1]."])

    answer = await _service(llm, QueryReranker(), self_check=False).answer("What is HNSW?")

    assert _steps(answer) == ["retrieve", "grade", "generate"]
    assert answer.self_check is None
    assert answer.grounded


async def test_grading_keeps_at_most_top_k_passages() -> None:
    llm = ScriptedLLM(generate=["x [1]."], self_check=[_verdict(True)])
    service = AgentAnswerService(
        FakeRetriever([make_chunk(str(i)) for i in range(8)]),
        QueryReranker(),
        llm,
        top_k=3,
        config=AgentConfig(),
    )

    await service.answer("q?")

    generate_prompt = llm.received[0][1].content
    assert "[3]" in generate_prompt
    assert "[4]" not in generate_prompt
