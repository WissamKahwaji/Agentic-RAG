"""Retrieve-then-generate: answer a question from retrieved chunks, with verified citations."""

import logging
import re
import time
import uuid
from collections.abc import Sequence
from typing import Literal, Protocol

from pydantic import BaseModel, Field

from agentic_rag.llm import ChatMessage, LLMProvider
from agentic_rag.prompts import GROUNDED_ANSWER, NO_ANSWER, grounded_answer_user_message
from agentic_rag.retrieval import RetrievedChunk, Retriever

logger = logging.getLogger(__name__)

# Matches [1], [12] and grouped forms such as [1, 3].
_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
# A closing code fence followed by markers on the same line: "``` [2]."
_FENCE_WITH_MARKERS = re.compile(
    r"^([ \t]*(?:```|~~~))[ \t]*((?:\[\d+(?:\s*,\s*\d+)*\])+)[ \t]*\.?[ \t]*$", re.MULTILINE
)


class Citation(BaseModel):
    number: int = Field(description="The marker used in the answer text, e.g. 2 for [2].")
    chunk_id: uuid.UUID
    source: str
    title: str
    location: str
    page: int | None
    section: str | None
    content: str
    score: float


class Timings(BaseModel):
    retrieval_ms: int = Field(description="Query rewriting, search, and grading.")
    generation_ms: int = Field(description="Writing the answer and checking it.")


class TraceStep(BaseModel):
    """One step the pipeline took, for showing how an answer was produced."""

    step: Literal[
        "condense", "retrieve", "grade", "reformulate", "generate", "self_check", "refuse"
    ]
    ms: int
    detail: str


class SelfCheck(BaseModel):
    passed: bool
    unsupported_claims: list[str] = Field(default_factory=list)


class HistoryTurn(BaseModel):
    question: str = Field(max_length=2000)
    answer: str = Field(max_length=8000)


class Answer(BaseModel):
    answer: str
    grounded: bool = Field(description="True when the answer cites at least one valid source.")
    refused: bool = Field(description="True when the sources did not contain an answer.")
    citations: list[Citation]
    model: str
    prompt: str = Field(description="Prompt name and version, e.g. grounded_answer@v1.")
    timings: Timings
    search_query: str | None = Field(
        default=None, description="The query actually searched, if it was rewritten."
    )
    self_check: SelfCheck | None = Field(
        default=None, description="Verdict of the answer self-check, when it ran."
    )
    trace: list[TraceStep] = Field(default_factory=list)


class QuestionAnswerer(Protocol):
    async def answer(
        self, question: str, top_k: int | None = None, history: Sequence[HistoryTurn] = ()
    ) -> Answer: ...


class AnswerService:
    def __init__(
        self,
        retriever: Retriever,
        llm: LLMProvider,
        top_k: int,
        temperature: float,
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._top_k = top_k
        self._temperature = temperature

    async def answer(
        self, question: str, top_k: int | None = None, history: Sequence[HistoryTurn] = ()
    ) -> Answer:
        """Single pass: retrieve, then generate. `history` is ignored (no query rewriting)."""
        started = time.perf_counter()
        chunks = await self._retriever.search(question, top_k or self._top_k)
        retrieved = time.perf_counter()

        if not chunks:
            text = NO_ANSWER
        else:
            result = await self._llm.chat(
                [
                    ChatMessage("system", GROUNDED_ANSWER.system),
                    ChatMessage("user", grounded_answer_user_message(question, chunks)),
                ],
                temperature=self._temperature,
            )
            text = result.text.strip()
        generated = time.perf_counter()

        text, citations, refused = finalize_answer(text, chunks)
        return Answer(
            answer=text,
            grounded=bool(citations) and not refused,
            refused=refused,
            citations=citations,
            model=self._llm.model,
            prompt=f"{GROUNDED_ANSWER.name}@v{GROUNDED_ANSWER.version}",
            timings=Timings(
                retrieval_ms=round((retrieved - started) * 1000),
                generation_ms=round((generated - retrieved) * 1000),
            ),
        )


def finalize_answer(text: str, chunks: list[RetrievedChunk]) -> tuple[str, list[Citation], bool]:
    """Verify citations and detect refusals. Returns (text, citations, refused)."""
    text, citations = resolve_citations(normalize_code_fences(text), chunks)
    refused = is_refusal(text)
    if refused:
        # Models often append "[1][2]..." to a refusal; citing sources for "I don't know"
        # would mislead the reader, so drop the markers along with the citations.
        text, citations = _tidy(_CITATION.sub("", text)).strip(), []
    logger.info("answered: citations=%s refused=%s", [c.number for c in citations], refused)
    return text, citations, refused


def normalize_code_fences(text: str) -> str:
    """Move citation markers off a closing code fence onto their own line.

    Asked to cite code, models often write "``` [2]". Markdown does not treat a fence line with
    trailing text as closing, so everything after it would render as code.
    """
    return _FENCE_WITH_MARKERS.sub(r"\1\n\n\2", text)


def resolve_citations(text: str, chunks: list[RetrievedChunk]) -> tuple[str, list[Citation]]:
    """Map `[n]` markers to retrieved chunks; strip markers that point at no source.

    Small models sometimes cite a number that was never provided. Leaving it in would show the
    user a citation that cannot be checked, which is worse than no citation.
    """
    cited: dict[int, Citation] = {}

    def replace(match: re.Match[str]) -> str:
        numbers = [int(n) for n in match.group(1).split(",")]
        valid = [n for n in numbers if 1 <= n <= len(chunks)]
        for n in valid:
            if n not in cited:
                cited[n] = _citation(n, chunks[n - 1])
        if len(valid) != len(numbers):
            logger.warning("dropped invalid citation(s) %s", set(numbers) - set(valid))
        return "".join(f"[{n}]" for n in valid)

    return _tidy(_CITATION.sub(replace, text)), list(cited.values())


def _tidy(text: str) -> str:
    """Remove the whitespace left behind by stripped markers: "word [9]." -> "word."."""
    return re.sub(r"[ \t]+([.,;:])", r"\1", re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE))


def is_refusal(text: str) -> bool:
    # Normalise before matching: models put markers inside words ("I don[1]'t know") and mix
    # typographic and straight apostrophes.
    normalised = _CITATION.sub("", text).replace("\u2019", "'").lower()
    return NO_ANSWER.lower().rstrip(".") in " ".join(normalised.split())


def _citation(number: int, chunk: RetrievedChunk) -> Citation:
    return Citation(
        number=number,
        chunk_id=chunk.chunk_id,
        source=chunk.source,
        title=chunk.title,
        location=chunk.location,
        page=chunk.page,
        section=chunk.section,
        content=chunk.content,
        score=round(chunk.score, 4),
    )
