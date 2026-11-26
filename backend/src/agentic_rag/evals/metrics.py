"""Retrieval metrics over a labelled ("golden") question set.

A retrieved chunk is relevant when it comes from a labelled source *and* location: a heading
path (matched by prefix, so a label on "pgvector > HNSW" also accepts "pgvector > HNSW > Index
Options") or a PDF page.
"""

from dataclasses import dataclass, field
from statistics import mean, median

from pydantic import BaseModel

from agentic_rag.retrieval import RetrievedChunk


class RelevantSpan(BaseModel):
    source: str
    section: str | None = None
    pages: list[int] | None = None

    def matches(self, chunk: RetrievedChunk) -> bool:
        if chunk.source != self.source:
            return False
        if self.pages is not None:
            return chunk.page in self.pages
        if self.section is not None:
            return chunk.section is not None and (
                chunk.section == self.section or chunk.section.startswith(self.section + " > ")
            )
        return True


class GoldenQuestion(BaseModel):
    id: str
    kind: str  # "keyword": hinges on exact terms/identifiers; "natural": paraphrased wording
    question: str
    relevant: list[RelevantSpan]


def first_relevant_rank(question: GoldenQuestion, results: list[RetrievedChunk]) -> int | None:
    """1-based rank of the first relevant chunk, or None if none was retrieved."""
    for rank, chunk in enumerate(results, start=1):
        if any(span.matches(chunk) for span in question.relevant):
            return rank
    return None


@dataclass
class Summary:
    n: int
    hit_at_1: float
    hit_at_5: float
    mrr_at_10: float
    p50_ms: float
    misses: list[str] = field(default_factory=list)


def summarize(ranks: dict[str, int | None], latencies_ms: list[float]) -> Summary:
    values = list(ranks.values())
    return Summary(
        n=len(values),
        hit_at_1=mean(r is not None and r <= 1 for r in values),
        hit_at_5=mean(r is not None and r <= 5 for r in values),
        mrr_at_10=mean(1 / r if r is not None and r <= 10 else 0.0 for r in values),
        p50_ms=median(latencies_ms) if latencies_ms else 0.0,
        misses=[qid for qid, r in ranks.items() if r is None or r > 5],
    )
