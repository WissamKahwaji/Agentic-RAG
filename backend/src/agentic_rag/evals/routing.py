"""Routing eval: `uv run python -m agentic_rag.evals.routing` (or `make eval-routing`).

The hybrid router runs only after the documents failed, so every question here is one the
documents cannot answer, labelled with the acceptable fallback(s): "web" or "general". The
costly error is answering a factual question from model memory ("general" when "web" is
expected), because that answer has no sources.
"""

import asyncio
import json
from pathlib import Path

import httpx

from agentic_rag.agent.graph import classify_route
from agentic_rag.config import get_settings
from agentic_rag.llm import OllamaProvider

ROUTING = Path(__file__).resolve().parents[3] / "evals" / "routing.jsonl"


async def main() -> None:
    settings = get_settings()
    rows = [json.loads(line) for line in ROUTING.read_text().splitlines() if line.strip()]
    correct = 0
    unsourced_facts: list[str] = []
    async with httpx.AsyncClient(timeout=settings.http_timeout_s) as http:
        llm = OllamaProvider(
            http, settings.ollama_base_url, settings.llm_model, settings.llm_num_ctx
        )
        for row in rows:
            decision = await classify_route(llm, row["question"], settings.corpus_description)
            ok = decision.route in row["acceptable"]
            correct += ok
            if row["acceptable"] == ["web"] and decision.route == "general":
                unsourced_facts.append(row["id"])
            mark = "ok  " if ok else "MISS"
            print(f"  {mark} {row['id']:<24} -> {decision.route:<9} ({decision.reason[:70]})")
    n = len(rows)
    print(f"\nrouting accuracy {correct}/{n} ({correct / n:.0%})")
    print(f"factual questions answered from memory: {len(unsourced_facts)} {unsourced_facts or ''}")


if __name__ == "__main__":
    asyncio.run(main())
