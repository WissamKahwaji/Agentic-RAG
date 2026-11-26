"""Prompt templates. Each prompt is named and versioned; bump the version on any wording change
so eval results can be attributed to a specific prompt."""

from dataclasses import dataclass

from agentic_rag.retrieval import RetrievedChunk

NO_ANSWER = "I don't know based on the provided documents."


@dataclass(frozen=True)
class Prompt:
    name: str
    version: int
    system: str


GROUNDED_ANSWER = Prompt(
    name="grounded_answer",
    # v2: adds a worked example and repeats the citation rule after the question. v1 skipped
    # citations on code-heavy "how do I" answers (cited in 13/18 runs on the sample corpus).
    version=2,
    system=f"""You answer questions using ONLY the numbered sources provided by the user.

Rules:
- Cite the source for every fact: put its number in square brackets at the end of the sentence
  or bullet, e.g. [1] or [2][3]. When you show code or SQL, cite the source right after it.
- Only cite numbers that appear in the sources.
- Do not use any knowledge that is not in the sources, even if you know the answer.
- If the sources do not contain the answer, reply with exactly: {NO_ANSWER}
- Be concise and direct. Use short paragraphs or bullet points. Keep code exactly as written in
  the sources.

Example of the required style:
Use the `--reload` flag during development so the server restarts on code changes [2].
```bash
uvicorn main:app --reload
```
This command comes from the tutorial [2]. Do not use it in production [4].""",
)

CITATION_REMINDER = "Answer using only the sources above, and cite them like [1]."


def format_sources(chunks: list[RetrievedChunk]) -> str:
    """Number sources from 1; the numbers are what the model cites."""
    blocks = []
    for number, chunk in enumerate(chunks, start=1):
        header = f"[{number}] {chunk.title} ({chunk.location})"
        blocks.append(f"{header}\n{chunk.content}")
    return "\n\n---\n\n".join(blocks)


def grounded_answer_user_message(question: str, chunks: list[RetrievedChunk]) -> str:
    # Repeating the key instruction last exploits recency: small models weight the end of the
    # prompt most heavily, and the sources can push the system prompt thousands of tokens back.
    return (
        f"Sources:\n\n{format_sources(chunks)}\n\n===\n\n"
        f"Question: {question}\n\n{CITATION_REMINDER}"
    )
