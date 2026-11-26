"""Terminal client for the running API: `uv run python -m agentic_rag.ask_cli "question"`."""

import argparse
import sys
import textwrap

import httpx


def main() -> int:
    parser = argparse.ArgumentParser(description="Ask the running Agentic RAG API a question.")
    parser.add_argument("question")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args()

    try:
        resp = httpx.post(
            f"{args.url}/ask",
            json={"question": args.question, "top_k": args.top_k},
            timeout=180,
        )
    except httpx.ConnectError:
        print(f"Cannot reach {args.url}. Is the API running? Start it with `make dev`.")
        return 1
    if resp.status_code != 200:
        print(f"HTTP {resp.status_code}: {resp.text}")
        return 1

    body = resp.json()
    print(f"\n{body['answer']}\n")
    for c in body["citations"]:
        snippet = textwrap.shorten(c["content"], width=110, placeholder=" …")
        print(f"  [{c['number']}] {c['source']} ({c['location']}), similarity {c['score']:.2f}")
        print(f"      {snippet}")
    t = body["timings"]
    print(
        f"\ngrounded={body['grounded']}  model={body['model']}  prompt={body['prompt']}  "
        f"retrieval={t['retrieval_ms']}ms  generation={t['generation_ms']}ms"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
