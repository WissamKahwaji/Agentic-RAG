"""The agentic RAG graph (LangGraph).

Documents mode (default):

    START -> condense -> retrieve -> grade --(relevant passages)--> generate -> self_check -> END
                            ^          |                               ^             |
                            |          +--(none, retries left)--> reformulate        |
                            |          +--(none, no retries)----> refuse -> END      |
                            +--------------------------------------------------------+
                                       (self_check failed, attempts left: regenerate)

Hybrid mode adds Corrective RAG: when the documents cannot answer (no passage passes grading,
or the grounded answer declines), a router picks the fallback:

    ... grade --(no relevant passages)--> route --web-----> web_search -> grade -> generate | refuse
    ... generate --(declined)-----------> route --general-> answer_direct -> END

Documents always come first by construction: every question is searched in the documents, and
the router only runs after they failed. (An earlier design routed *before* retrieval; it sent
2 of 9 documents questions to the web, e.g. "How do I speed up HNSW index builds?".)

Every loop is bounded by a settings value, so the graph always
terminates. Nodes are small async functions that return only the state keys they change.
"""

import logging
import operator
import time
import uuid
from dataclasses import dataclass, replace
from typing import Annotated, Literal, TypedDict
from urllib.parse import urlparse

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from agentic_rag.agent.verify import unsupported_claims
from agentic_rag.answering import AnswerSource, Knowledge, SelfCheck, TraceStep, is_refusal
from agentic_rag.llm import ChatMessage, LLMProvider, chat_structured
from agentic_rag.prompts import (
    CONDENSE_QUESTION,
    DIRECT_ANSWER,
    GROUNDED_ANSWER,
    NO_ANSWER,
    REFORMULATE_QUERY,
    WEB_GROUNDED_ANSWER,
    condense_user_message,
    grounded_answer_user_message,
    reformulate_user_message,
    revision_note,
    route_system_prompt,
)
from agentic_rag.reranking import Reranker
from agentic_rag.retrieval import RetrievedChunk, Retriever
from agentic_rag.web import WebResult, WebSearch

logger = logging.getLogger(__name__)


Route = Literal["web", "general"]


class AgentState(TypedDict, total=False):
    question: str
    history: list[tuple[str, str]]
    top_k: int
    knowledge: Knowledge

    route: Route
    source: AnswerSource  # where `relevant` came from: documents or web
    web_searched: bool

    search_query: str
    tried_queries: list[str]
    candidates: list[RetrievedChunk]
    relevant: list[RetrievedChunk]

    answer: str
    generations: int
    self_check: SelfCheck | None

    # Each node appends its own steps; the reducer concatenates instead of overwriting.
    trace: Annotated[list[TraceStep], operator.add]


@dataclass(frozen=True)
class AgentConfig:
    candidates: int = 10
    relevance_threshold: float = 0.05
    max_retrieval_attempts: int = 2
    max_generation_attempts: int = 2
    self_check: bool = True
    temperature: float = 0.1
    web_results: int = 5
    corpus_description: str = "the user's uploaded documents"


class _Query(BaseModel):
    query: str = Field(min_length=2, max_length=300)


class _Route(BaseModel):
    route: Route
    reason: str = Field(default="", max_length=300)


class RouteDecision(BaseModel):
    route: Route
    reason: str = ""


async def classify_route(llm: LLMProvider, question: str, corpus_description: str) -> RouteDecision:
    """The router's decision for one question (shared by the graph and the routing eval)."""
    result = await chat_structured(
        llm,
        [
            ChatMessage("system", route_system_prompt(corpus_description)),
            ChatMessage("user", question),
        ],
        _Route,
    )
    return RouteDecision(route=result.route, reason=result.reason)


def web_result_to_chunk(result: WebResult) -> RetrievedChunk:
    """Represent a web result as a passage, so grading, prompting and citations are shared."""
    return RetrievedChunk(
        chunk_id=uuid.uuid5(uuid.NAMESPACE_URL, result.url),
        source=result.url,
        title=result.title or urlparse(result.url).netloc,
        page=None,
        section=urlparse(result.url).netloc,
        content=result.content,
        score=0.0,
    )


def _ms(started: float) -> int:
    return round((time.perf_counter() - started) * 1000)


def build_agent_graph(
    retriever: Retriever,
    reranker: Reranker,
    llm: LLMProvider,
    config: AgentConfig,
    web: WebSearch | None = None,
) -> CompiledStateGraph[AgentState, None, AgentState, AgentState]:
    # ------------------------------------------------------------------ nodes

    async def condense(state: AgentState) -> AgentState:
        """Turn a follow-up ("what about IVFFlat?") into a standalone search query."""
        if not state.get("history"):
            return {"search_query": state["question"], "tried_queries": []}
        started = time.perf_counter()
        result = await chat_structured(
            llm,
            [
                ChatMessage("system", CONDENSE_QUESTION.system),
                ChatMessage("user", condense_user_message(state["question"], state["history"])),
            ],
            _Query,
        )
        step = TraceStep(step="condense", ms=_ms(started), detail=f"Searched for: {result.query}")
        return {"search_query": result.query, "tried_queries": [], "trace": [step]}

    async def retrieve(state: AgentState) -> AgentState:
        started = time.perf_counter()
        candidates = await retriever.search(state["search_query"], config.candidates)
        detail = f"{len(candidates)} candidate passages for “{state['search_query']}”"
        return {
            "candidates": candidates,
            "source": "documents",
            "tried_queries": [*state.get("tried_queries", []), state["search_query"]],
            "trace": [TraceStep(step="retrieve", ms=_ms(started), detail=detail)],
        }

    async def route(state: AgentState) -> AgentState:
        """The documents could not answer: choose web search or model knowledge (hybrid only)."""
        started = time.perf_counter()
        result = await classify_route(llm, state["search_query"], config.corpus_description)
        detail = f"{result.route}: {result.reason}" if result.reason else result.route
        return {
            "route": result.route,
            "trace": [TraceStep(step="route", ms=_ms(started), detail=detail)],
        }

    async def web_search(state: AgentState) -> AgentState:
        started = time.perf_counter()
        results = await web.search(state["search_query"], config.web_results) if web else []
        detail = f"{len(results)} web results for “{state['search_query']}”"
        return {
            "candidates": [web_result_to_chunk(r) for r in results],
            "source": "web",
            "web_searched": True,
            # A fresh draft: the documents-path answer (if any) must not leak into this one.
            "generations": 0,
            "self_check": None,
            "trace": [TraceStep(step="web_search", ms=_ms(started), detail=detail)],
        }

    async def answer_direct(state: AgentState) -> AgentState:
        """General-knowledge answer with no sources. Labelled as such; not self-checked."""
        started = time.perf_counter()
        result = await llm.chat(
            [
                ChatMessage("system", DIRECT_ANSWER.system),
                ChatMessage("user", state["question"]),
            ],
            temperature=config.temperature,
        )
        return {
            "answer": result.text.strip(),
            "relevant": [],
            "source": "model",
            "self_check": None,
            "trace": [
                TraceStep(
                    step="answer_direct", ms=_ms(started), detail="Answered from model knowledge"
                )
            ],
        }

    async def grade(state: AgentState) -> AgentState:
        """Score each candidate with the cross-encoder; keep those above the threshold."""
        started = time.perf_counter()
        candidates = state["candidates"]
        scores = await reranker.score(
            state["search_query"], [f"{c.title} > {c.location}\n{c.content}" for c in candidates]
        )
        ranked = sorted(zip(candidates, scores, strict=True), key=lambda p: p[1], reverse=True)
        relevant = [replace(c, score=s) for c, s in ranked if s >= config.relevance_threshold][
            : state["top_k"]
        ]
        dropped = len(candidates) - len(relevant)
        detail = (
            f"kept {len(relevant)} relevant, dropped {dropped} below {config.relevance_threshold}"
        )
        return {
            "relevant": relevant,
            "trace": [TraceStep(step="grade", ms=_ms(started), detail=detail)],
        }

    async def reformulate(state: AgentState) -> AgentState:
        started = time.perf_counter()
        result = await chat_structured(
            llm,
            [
                ChatMessage("system", REFORMULATE_QUERY.system),
                ChatMessage(
                    "user", reformulate_user_message(state["question"], state["tried_queries"])
                ),
            ],
            _Query,
            temperature=0.4,  # a little variety: the point is to try something different
        )
        step = TraceStep(step="reformulate", ms=_ms(started), detail=f"Retrying as: {result.query}")
        return {"search_query": result.query, "trace": [step]}

    async def refuse(state: AgentState) -> AgentState:
        tries = len(state.get("tried_queries", []))
        detail = f"No relevant passages after {tries} search{'es' if tries > 1 else ''}"
        return {
            "answer": NO_ANSWER,
            "relevant": [],
            "self_check": None,
            "trace": [TraceStep(step="refuse", ms=0, detail=detail)],
        }

    async def generate(state: AgentState) -> AgentState:
        started = time.perf_counter()
        prompt = WEB_GROUNDED_ANSWER if state.get("source") == "web" else GROUNDED_ANSWER
        messages = [
            ChatMessage("system", prompt.system),
            ChatMessage("user", grounded_answer_user_message(state["question"], state["relevant"])),
        ]
        check = state.get("self_check")
        if check is not None and not check.passed:
            # Regenerating after a failed self-check: show the model its draft and the problems.
            messages += [
                ChatMessage("assistant", state["answer"]),
                ChatMessage("user", revision_note(check.unsupported_claims)),
            ]
        result = await llm.chat(messages, temperature=config.temperature)
        attempt = state.get("generations", 0) + 1
        detail = "Wrote the answer" if attempt == 1 else f"Revised the answer (attempt {attempt})"
        return {
            "answer": result.text.strip(),
            "generations": attempt,
            "trace": [TraceStep(step="generate", ms=_ms(started), detail=detail)],
        }

    async def self_check(state: AgentState) -> AgentState:
        """Verify each cited claim against the sources it cites (see agent/verify.py)."""
        if is_refusal(state["answer"]):
            return {"self_check": None}
        started = time.perf_counter()
        claims = await unsupported_claims(llm, state["answer"], state["relevant"])
        detail = (
            "Every claim is supported by its sources"
            if not claims
            else f"{len(claims)} unsupported claim{'s' if len(claims) != 1 else ''} found"
        )
        return {
            "self_check": SelfCheck(passed=not claims, unsupported_claims=claims),
            "trace": [TraceStep(step="self_check", ms=_ms(started), detail=detail)],
        }

    # ----------------------------------------------------------------- routing

    def can_fall_back(state: AgentState) -> bool:
        """Hybrid mode, the documents failed, and no fallback has been tried yet."""
        return (
            state.get("knowledge") == "hybrid"
            and web is not None
            and state.get("source") == "documents"
            and "route" not in state
        )

    def after_route(state: AgentState) -> Literal["web_search", "answer_direct"]:
        return "web_search" if state["route"] == "web" else "answer_direct"

    def after_grade(state: AgentState) -> Literal["generate", "reformulate", "route", "refuse"]:
        if state["relevant"]:
            return "generate"
        if can_fall_back(state):
            return "route"  # Corrective RAG: better sources beat another document query
        if state.get("source") == "documents" and (
            len(state["tried_queries"]) < config.max_retrieval_attempts
        ):
            return "reformulate"
        return "refuse"

    def after_generate(state: AgentState) -> Literal["self_check", "route", "__end__"]:
        if is_refusal(state["answer"]) and can_fall_back(state):
            # Passages were on topic but did not answer (e.g. "does pgvector support DiskANN?").
            return "route"
        return "self_check" if config.self_check else "__end__"

    def after_check(state: AgentState) -> Literal["generate", "__end__"]:
        check = state.get("self_check")
        failed = check is not None and not check.passed
        if failed and state["generations"] < config.max_generation_attempts:
            return "generate"
        return "__end__"  # LangGraph's END constant, spelled out for the Literal return type

    # ------------------------------------------------------------------- graph

    graph = StateGraph(AgentState)
    graph.add_node("condense", condense)
    graph.add_node("route", route)
    graph.add_node("web_search", web_search)
    graph.add_node("answer_direct", answer_direct)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade", grade)
    graph.add_node("reformulate", reformulate)
    graph.add_node("refuse", refuse)
    graph.add_node("generate", generate)
    graph.add_node("self_check", self_check)

    graph.add_edge(START, "condense")
    graph.add_edge("condense", "retrieve")
    graph.add_conditional_edges("route", after_route)
    graph.add_edge("web_search", "grade")
    graph.add_edge("answer_direct", END)
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges("grade", after_grade)
    graph.add_edge("reformulate", "retrieve")
    graph.add_edge("refuse", END)
    graph.add_conditional_edges("generate", after_generate)
    graph.add_conditional_edges("self_check", after_check)
    return graph.compile()
