"""Runs the agent graph and shapes its final state into the public `Answer`."""

from collections.abc import Sequence

from agentic_rag.agent.graph import AgentConfig, AgentState, build_agent_graph
from agentic_rag.answering import Answer, HistoryTurn, Timings, finalize_answer
from agentic_rag.llm import LLMProvider
from agentic_rag.prompts import GROUNDED_ANSWER
from agentic_rag.reranking import Reranker
from agentic_rag.retrieval import Retriever

_RETRIEVAL_STEPS = {"condense", "retrieve", "grade", "reformulate", "refuse"}
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
    ) -> None:
        self._graph = build_agent_graph(retriever, reranker, llm, config)
        self._model = llm.model
        self._top_k = top_k

    async def answer(
        self, question: str, top_k: int | None = None, history: Sequence[HistoryTurn] = ()
    ) -> Answer:
        initial: AgentState = {
            "question": question,
            "history": [(t.question, t.answer) for t in history],
            "top_k": top_k or self._top_k,
            "trace": [],
        }
        state = await self._graph.ainvoke(initial, {"recursion_limit": _RECURSION_LIMIT})

        relevant = state.get("relevant", [])
        text, citations, refused = finalize_answer(state["answer"], relevant)
        trace = state["trace"]
        check = state.get("self_check")
        return Answer(
            answer=text,
            grounded=bool(citations) and not refused and (check is None or check.passed),
            refused=refused,
            citations=citations,
            model=self._model,
            prompt=f"{GROUNDED_ANSWER.name}@v{GROUNDED_ANSWER.version}",
            timings=Timings(
                retrieval_ms=sum(s.ms for s in trace if s.step in _RETRIEVAL_STEPS),
                generation_ms=sum(s.ms for s in trace if s.step not in _RETRIEVAL_STEPS),
            ),
            search_query=state["search_query"] if state["search_query"] != question else None,
            self_check=None if refused else check,
            trace=trace,
        )
