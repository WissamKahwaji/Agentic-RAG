"""Runs the agent graph and shapes its final state into the public `Answer`."""

from collections.abc import Sequence

from agentic_rag.agent.graph import AgentConfig, AgentState, build_agent_graph
from agentic_rag.answering import Answer, HistoryTurn, Knowledge, Timings, finalize_answer
from agentic_rag.llm import LLMProvider
from agentic_rag.prompts import DIRECT_ANSWER, GROUNDED_ANSWER, WEB_GROUNDED_ANSWER
from agentic_rag.reranking import Reranker
from agentic_rag.retrieval import Retriever
from agentic_rag.web import WebSearch

_RETRIEVAL_STEPS = {"condense", "route", "retrieve", "web_search", "grade", "reformulate", "refuse"}
# Generous: the bounded loops need at most ~12 steps; this only guards against a wiring bug.
_RECURSION_LIMIT = 30


class AgentAnswerService:
    def __init__(
        self,
        retriever: Retriever,
        reranker: Reranker,
        llm: LLMProvider,
        top_k: int,
        config: AgentConfig,
        web: WebSearch | None = None,
    ) -> None:
        self._graph = build_agent_graph(retriever, reranker, llm, config, web)
        self._model = llm.model
        self._top_k = top_k

    async def answer(
        self,
        question: str,
        top_k: int | None = None,
        history: Sequence[HistoryTurn] = (),
        knowledge: Knowledge = "documents",
    ) -> Answer:
        initial: AgentState = {
            "question": question,
            "knowledge": knowledge,
            "history": [(t.question, t.answer) for t in history],
            "top_k": top_k or self._top_k,
            "trace": [],
        }
        state = await self._graph.ainvoke(initial, {"recursion_limit": _RECURSION_LIMIT})

        relevant = state.get("relevant", [])
        text, citations, refused = finalize_answer(state["answer"], relevant)
        trace = state["trace"]
        check = state.get("self_check")
        source = "none" if refused else state.get("source", "documents")
        prompt = {"web": WEB_GROUNDED_ANSWER, "model": DIRECT_ANSWER}.get(source, GROUNDED_ANSWER)
        return Answer(
            answer=text,
            grounded=bool(citations) and not refused and (check is None or check.passed),
            refused=refused,
            citations=citations,
            source=source,
            model=self._model,
            prompt=f"{prompt.name}@v{prompt.version}",
            timings=Timings(
                retrieval_ms=sum(s.ms for s in trace if s.step in _RETRIEVAL_STEPS),
                generation_ms=sum(s.ms for s in trace if s.step not in _RETRIEVAL_STEPS),
            ),
            search_query=state["search_query"] if state["search_query"] != question else None,
            self_check=None if refused else check,
            trace=trace,
            contexts=[c.content for c in relevant],
        )
