"""Claim-level verification of a cited answer.

Asking a small model to audit a whole answer at once ("list every unsupported claim") is a hard
task, and a 7B model does it badly: in our eval it flagged correct claims on most answers. This
splits the answer into its cited claims and asks one narrow yes/no question per claim, showing
only the sources that claim cites. Narrower questions are much easier for small models.
"""

import asyncio
import re
from dataclasses import dataclass

from pydantic import BaseModel

from agentic_rag.llm import ChatMessage, LLMProvider, StructuredOutputError, chat_structured
from agentic_rag.prompts import VERIFY_CLAIM, verify_claim_user_message
from agentic_rag.retrieval import RetrievedChunk

_MARKER = re.compile(r"\[(\d+)\]")
_FENCE = re.compile(r"```.*?(?:```|$)", re.DOTALL)
# Sentence ends, or line breaks (list items and paragraphs are separate claims).
_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z`*\[(])|\n+")
MIN_CLAIM_CHARS = 20


@dataclass(frozen=True)
class Claim:
    text: str  # without citation markers
    cites: tuple[int, ...]


def split_claims(answer: str) -> list[Claim]:
    """Cited, factual sentences of an answer. Code blocks and uncited lead-ins are skipped."""
    claims: list[Claim] = []
    for raw in _SPLIT.split(_FENCE.sub("\n", answer)):
        cites = tuple(dict.fromkeys(int(n) for n in _MARKER.findall(raw)))
        text = " ".join(_MARKER.sub("", raw).strip(" -*\t").split())
        text = re.sub(r"\s+([.,;:!?])", r"\1", text)
        if cites and len(text) >= MIN_CLAIM_CHARS:
            claims.append(Claim(text=text, cites=cites))
    return claims


class _Support(BaseModel):
    supported: bool


async def unsupported_claims(
    llm: LLMProvider, answer: str, sources: list[RetrievedChunk], concurrency: int = 2
) -> list[str]:
    """Return the claims whose cited sources do not support them (empty list = all good).

    Checks run with bounded concurrency: firing every claim at once bursts past hosted free-tier
    rate limits (Groq: 8k tokens/minute) and turned one answer into a cascade of HTTP 429s.
    """
    gate = asyncio.Semaphore(concurrency)

    async def check(claim: Claim) -> str | None:
        async with gate:
            return await _check(claim)

    async def _check(claim: Claim) -> str | None:
        cited = [sources[n - 1] for n in claim.cites if 1 <= n <= len(sources)]
        if not cited:
            return claim.text
        try:
            verdict = await chat_structured(
                llm,
                [
                    ChatMessage("system", VERIFY_CLAIM.system),
                    ChatMessage("user", verify_claim_user_message(claim.text, cited)),
                ],
                _Support,
            )
        except StructuredOutputError:
            return None  # inconclusive: don't accuse a claim we could not check
        return None if verdict.supported else claim.text

    results = await asyncio.gather(*(check(c) for c in split_claims(answer)))
    return [r for r in results if r is not None]
