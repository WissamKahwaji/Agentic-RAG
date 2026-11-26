"""The evaluation suite: `uv run python -m agentic_rag.evals.suite` (or `make eval`).

One run scores the whole system on the versioned dataset and writes a report:

  retrieval  hit@1, hit@5, MRR@10 of the configured retrieval pipeline (deterministic)
  behaviour  answered / declined / self-check flags / latency, per answer pipeline
  quality    Ragas metrics, judged by an LLM that should differ from the generator:
             - faithfulness:        are the answer's claims supported by the given passages?
             - answer_relevancy:    does the answer address the question?
             - context_precision:   are the passages useful for the reference answer, ranked well?
             - context_recall:      do the passages contain what the reference answer needs?
             - factual_correctness: does the answer agree with the reference answer (F1)?

Reports go to backend/evals/reports/ (JSON + Markdown, committed as history) and include a
config snapshot, so a score can always be traced to the code, prompts and models behind it.
Compare two runs with `--compare reports/<baseline>.json`.
"""

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median, quantiles
from typing import Any

import httpx
from pydantic_settings import BaseSettings, SettingsConfigDict

from agentic_rag import db, prompts
from agentic_rag.answering import Answer, QuestionAnswerer, strip_citations
from agentic_rag.config import REPO_ROOT, Settings, get_settings
from agentic_rag.embeddings import OllamaEmbedder
from agentic_rag.evals.grading import UNANSWERABLE, load_unanswerable
from agentic_rag.evals.metrics import GoldenQuestion, summarize
from agentic_rag.evals.retrieval import GOLDEN, evaluate, load_golden
from agentic_rag.llm import OllamaProvider
from agentic_rag.main import build_answer_service
from agentic_rag.reranking import CrossEncoderReranker
from agentic_rag.search import build_retriever

os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")  # no telemetry from the eval library

REPORTS = Path(__file__).resolve().parents[3] / "evals" / "reports"
JUDGED_METRICS = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "factual_correctness",
)


class EvalSettings(BaseSettings):
    """The judge model. Any OpenAI-compatible endpoint works (Ollama's /v1, Groq, ...)."""

    model_config = SettingsConfigDict(
        env_prefix="EVAL_", env_file=REPO_ROOT / ".env", extra="ignore"
    )

    judge_model: str = "qwen2.5:7b"
    judge_base_url: str = "http://localhost:11434/v1"
    judge_api_key: str = "ollama"
    judge_embedding_model: str = "bge-m3"
    judge_embedding_base_url: str = "http://localhost:11434/v1"
    concurrency: int = 2


# ------------------------------------------------------------------------------------ judging


@dataclass
class Judge:
    """Ragas metrics bound to the judge model."""

    metrics: dict[str, Callable[..., Awaitable[Any]]]

    @classmethod
    def create(cls, cfg: EvalSettings) -> "Judge":
        from openai import AsyncOpenAI
        from ragas.cache import DiskCacheBackend
        from ragas.embeddings.base import embedding_factory
        from ragas.llms import llm_factory
        from ragas.metrics.collections import (
            AnswerRelevancy,
            ContextPrecisionWithReference,
            ContextRecall,
            FactualCorrectness,
            Faithfulness,
        )

        # Cached judge calls make re-scoring unchanged answers nearly free.
        cache = DiskCacheBackend(cache_dir=str(REPORTS.parent / ".judge-cache"))
        llm = llm_factory(
            cfg.judge_model,
            client=AsyncOpenAI(base_url=cfg.judge_base_url, api_key=cfg.judge_api_key),
            cache=cache,
        )
        embeddings: Any = embedding_factory(
            "openai",
            model=cfg.judge_embedding_model,
            client=AsyncOpenAI(base_url=cfg.judge_embedding_base_url, api_key=cfg.judge_api_key),
        )
        faithfulness = Faithfulness(llm=llm)
        relevancy = AnswerRelevancy(llm=llm, embeddings=embeddings, strictness=1)
        precision = ContextPrecisionWithReference(llm=llm)
        recall = ContextRecall(llm=llm)
        correctness = FactualCorrectness(llm=llm, mode="f1")
        return cls(
            metrics={
                "faithfulness": lambda q, a, ctx, ref: faithfulness.ascore(
                    user_input=q, response=a, retrieved_contexts=ctx
                ),
                "answer_relevancy": lambda q, a, ctx, ref: relevancy.ascore(
                    user_input=q, response=a
                ),
                "context_precision": lambda q, a, ctx, ref: precision.ascore(
                    user_input=q, reference=ref, retrieved_contexts=ctx
                ),
                "context_recall": lambda q, a, ctx, ref: recall.ascore(
                    user_input=q, retrieved_contexts=ctx, reference=ref
                ),
                "factual_correctness": lambda q, a, ctx, ref: correctness.ascore(
                    response=a, reference=ref
                ),
            }
        )

    async def score(
        self, question: str, answer: Answer, reference: str
    ) -> tuple[dict[str, float | None], list[str]]:
        """Score one answer. Returns (scores, names of metrics whose judge call failed)."""
        text = strip_citations(answer.answer)
        scores: dict[str, float | None] = {}
        failures: list[str] = []
        for name, metric in self.metrics.items():
            if answer.refused and name in (
                "faithfulness",
                "answer_relevancy",
                "factual_correctness",
            ):
                # A refusal makes no claims (faithfulness is undefined) and misses the answer.
                scores[name] = None if name == "faithfulness" else 0.0
                continue
            if not answer.contexts and name.startswith("context_"):
                scores[name] = 0.0
                continue
            try:
                result = await metric(question, text, answer.contexts, reference)
                scores[name] = float(result.value)
            except Exception as exc:  # judge models sometimes return unparsable output
                scores[name] = None
                failures.append(f"{name}: {type(exc).__name__}")
        return scores, failures


# ------------------------------------------------------------------------------------ running


@dataclass
class ModeResult:
    mode: str
    rows: list[dict[str, Any]] = field(default_factory=list)


async def run_mode(
    mode: str,
    settings: Settings,
    service: QuestionAnswerer,
    golden: list[GoldenQuestion],
    unanswerable: list[tuple[str, str]],
    judge: Judge | None,
    concurrency: int,
) -> ModeResult:
    result = ModeResult(mode)
    gate = asyncio.Semaphore(concurrency)

    async def judged(
        q: GoldenQuestion, answer: Answer
    ) -> tuple[dict[str, float | None], list[str]]:
        if judge is None or q.reference is None:
            return {}, []
        async with gate:
            return await judge.score(q.question, answer, q.reference)

    pending: list[tuple[dict[str, Any], Awaitable[Any]]] = []
    for q in golden:
        started = time.perf_counter()
        answer = await service.answer(q.question)
        row = _row(q.id, q.question, True, answer, time.perf_counter() - started)
        pending.append((row, judged(q, answer)))
        print(f"  {mode:<8} {q.id:<24} {'declined' if answer.refused else 'answered'}", flush=True)
    for qid, question in unanswerable:
        started = time.perf_counter()
        answer = await service.answer(question)
        result.rows.append(_row(qid, question, False, answer, time.perf_counter() - started))
        print(f"  {mode:<8} {qid:<24} {'declined' if answer.refused else 'answered'}", flush=True)

    print(f"  {mode}: judging {len(pending)} answers…", flush=True)
    judged_rows = await asyncio.gather(*(task for _, task in pending))
    for (row, _), (scores, failures) in zip(pending, judged_rows, strict=True):
        row["scores"], row["judge_failures"] = scores, failures
        result.rows.append(row)
    return result


def _row(
    qid: str, question: str, answerable: bool, answer: Answer, seconds: float
) -> dict[str, Any]:
    return {
        "id": qid,
        "question": question,
        "answerable": answerable,
        "seconds": round(seconds, 2),
        "answer": answer.answer,
        "refused": answer.refused,
        "grounded": answer.grounded,
        "self_check_passed": None if answer.self_check is None else answer.self_check.passed,
        "cited": [f"{c.source} ({c.location})" for c in answer.citations],
        "trace": [s.step for s in answer.trace],
    }


# ------------------------------------------------------------------------------------ report


def summarize_mode(result: ModeResult) -> dict[str, Any]:
    answerable = [r for r in result.rows if r["answerable"]]
    unanswerable = [r for r in result.rows if not r["answerable"]]
    secs = sorted(r["seconds"] for r in result.rows)
    summary: dict[str, Any] = {
        "answered": _ratio(sum(not r["refused"] for r in answerable), len(answerable)),
        "declined_unanswerable": _ratio(sum(r["refused"] for r in unanswerable), len(unanswerable)),
        "self_check_flagged": _ratio(
            sum(r["self_check_passed"] is False for r in answerable), len(answerable)
        ),
        "latency_p50_s": round(median(secs), 2),
        "latency_p90_s": round(quantiles(secs, n=10)[-1], 2) if len(secs) > 1 else secs[0],
        "judge_failures": sum(len(r.get("judge_failures", [])) for r in answerable),
    }
    for metric in JUDGED_METRICS:
        values = [
            r["scores"][metric] for r in answerable if r.get("scores", {}).get(metric) is not None
        ]
        summary[metric] = {"mean": round(mean(values), 3) if values else None, "n": len(values)}
    return summary


def _ratio(n: int, total: int) -> dict[str, Any]:
    return {"value": round(n / total, 3) if total else None, "n": n, "of": total}


def config_snapshot(settings: Settings, cfg: EvalSettings, judged: bool) -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, cwd=REPO_ROOT
        ).stdout.strip()

    datasets = GOLDEN.read_bytes() + UNANSWERABLE.read_bytes()
    return {
        "git_sha": git("rev-parse", "--short", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain", "--", "backend/src")),
        "dataset_sha": hashlib.sha256(datasets).hexdigest()[:12],
        "llm": settings.llm_model,
        "embedding_model": settings.embedding_model,
        "reranker": settings.reranker_model,
        "retrieval_mode": settings.retrieval_mode,
        "relevance_threshold": settings.relevance_threshold,
        "top_k": settings.retrieval_top_k,
        "judge": cfg.judge_model if judged else None,
        "judge_is_generator": judged and cfg.judge_model == settings.llm_model,
        "prompts": {
            p.name: p.version for p in vars(prompts).values() if isinstance(p, prompts.Prompt)
        },
    }


def render_markdown(report: dict[str, Any], baseline: dict[str, Any] | None) -> str:
    cfg = report["config"]
    lines = [
        f"# Eval report {report['run_id']}",
        "",
        f"`{cfg['git_sha']}`{' (uncommitted changes)' if cfg['git_dirty'] else ''} · "
        f"dataset `{cfg['dataset_sha']}` · LLM `{cfg['llm']}` · "
        f"retrieval `{cfg['retrieval_mode']}` · judge `{cfg['judge']}`",
    ]
    if cfg["judge_is_generator"]:
        lines += ["", "> Judge = generator: quality scores are likely biased upward."]
    r = report["retrieval"]
    lines += [
        "",
        "## Retrieval",
        "",
        f"hit@1 **{r['hit_at_1']:.0%}** · hit@5 **{r['hit_at_5']:.0%}** · "
        f"MRR@10 **{r['mrr_at_10']:.3f}** · p50 {r['p50_ms']:.0f} ms",
        "",
        "## Answers",
        "",
        "| metric | " + " | ".join(report["modes"]) + " |",
        "| --- | " + " | ".join("---" for _ in report["modes"]) + " |",
    ]
    rows: list[tuple[str, Callable[[dict[str, Any]], float | None], str]] = [
        ("answered (answerable)", lambda s: s["answered"]["value"], "pct"),
        ("declined (unanswerable)", lambda s: s["declined_unanswerable"]["value"], "pct"),
        ("self-check flagged", lambda s: s["self_check_flagged"]["value"], "pct"),
        *[(m, (lambda s, m=m: s[m]["mean"]), "score") for m in JUDGED_METRICS],
        ("latency p50 (s)", lambda s: s["latency_p50_s"], "num"),
        ("latency p90 (s)", lambda s: s["latency_p90_s"], "num"),
    ]
    for label, get, fmt in rows:
        cells = []
        for mode, summary in report["modes"].items():
            value = get(summary)
            cell = _fmt(value, fmt)
            if baseline and mode in baseline.get("modes", {}):
                cell += _delta(value, get(baseline["modes"][mode]), fmt)
            cells.append(cell)
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    if baseline:
        lines += ["", f"Deltas are against `{baseline['run_id']}`."]
    failures = sum(s["judge_failures"] for s in report["modes"].values())
    if failures:
        lines += ["", f"{failures} judge calls failed and were excluded (see the JSON report)."]
    return "\n".join(lines) + "\n"


def _fmt(value: float | None, fmt: str) -> str:
    if value is None:
        return "n/a"
    return f"{value:.0%}" if fmt == "pct" else f"{value:.3f}" if fmt == "score" else f"{value:.1f}"


def _delta(value: float | None, base: float | None, fmt: str) -> str:
    if value is None or base is None or abs(value - base) < 1e-9:
        return ""
    d = value - base
    return f" ({d:+.0%})" if fmt == "pct" else f" ({d:+.3f})" if fmt == "score" else f" ({d:+.1f})"


# ------------------------------------------------------------------------------------ main


async def main(
    modes: list[str], limit: int | None, judge_enabled: bool, compare: Path | None
) -> None:
    settings = get_settings()
    cfg = EvalSettings()
    golden = load_golden()[:limit] if limit else load_golden()
    unanswerable = load_unanswerable()[:limit] if limit else load_unanswerable()
    judge = Judge.create(cfg) if judge_enabled else None

    reranker = CrossEncoderReranker(settings.reranker_model)
    await reranker.warm_up()
    engine = db.create_engine(str(settings.database_url))
    try:
        async with httpx.AsyncClient(timeout=settings.http_timeout_s) as http:
            sessions = db.create_session_factory(engine)
            embedder = OllamaEmbedder(
                http, settings.ollama_base_url, settings.embedding_model, settings.embedding_dim
            )
            llm = OllamaProvider(
                http, settings.ollama_base_url, settings.llm_model, settings.llm_num_ctx
            )

            retriever = build_retriever(
                settings.retrieval_mode, sessions, embedder, settings, reranker
            )
            ranks, latencies = await evaluate(retriever, golden)
            retrieval = summarize(ranks, latencies)

            results = []
            for mode in modes:
                mode_settings = settings.model_copy(update={"answer_mode": mode})
                service = await build_answer_service(
                    mode_settings, sessions, embedder, llm, reranker
                )
                results.append(
                    await run_mode(
                        mode, mode_settings, service, golden, unanswerable, judge, cfg.concurrency
                    )
                )
    finally:
        await engine.dispose()

    run_id = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    report: dict[str, Any] = {
        "run_id": run_id,
        "config": config_snapshot(settings, cfg, judge is not None),
        "limit": limit,
        "retrieval": {
            "hit_at_1": retrieval.hit_at_1,
            "hit_at_5": retrieval.hit_at_5,
            "mrr_at_10": retrieval.mrr_at_10,
            "p50_ms": retrieval.p50_ms,
            "missed_at_5": retrieval.misses,
        },
        "modes": {r.mode: summarize_mode(r) for r in results},
        "rows": {r.mode: r.rows for r in results},
    }
    path = await asyncio.to_thread(write_report, report, compare)
    print(f"report: {path.relative_to(REPO_ROOT)}")


def write_report(report: dict[str, Any], compare: Path | None) -> Path:
    baseline = json.loads(compare.read_text()) if compare else None
    REPORTS.mkdir(parents=True, exist_ok=True)
    stem = REPORTS / f"{report['run_id']}-{report['config']['git_sha']}"
    stem.with_suffix(".json").write_text(json.dumps(report, indent=2))
    markdown = render_markdown(report, baseline)
    stem.with_suffix(".md").write_text(markdown)
    print("\n" + markdown)
    return stem.with_suffix(".md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--mode", action="append", choices=["simple", "agentic"])
    parser.add_argument(
        "--limit", type=int, help="only the first N questions of each set (smoke run)"
    )
    parser.add_argument("--no-judge", action="store_true", help="skip LLM-judged metrics")
    parser.add_argument("--compare", type=Path, help="baseline report JSON to show deltas against")
    args = parser.parse_args()
    asyncio.run(
        main(args.mode or ["simple", "agentic"], args.limit, not args.no_judge, args.compare)
    )
