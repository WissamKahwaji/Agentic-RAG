from types import SimpleNamespace
from typing import Any

from agentic_rag.answering import Answer, Timings
from agentic_rag.evals.suite import (
    JUDGED_METRICS,
    Judge,
    ModeResult,
    render_markdown,
    summarize_mode,
)


def _answer(text: str, refused: bool = False, contexts: list[str] | None = None) -> Answer:
    return Answer(
        answer=text,
        grounded=not refused,
        refused=refused,
        citations=[],
        model="m",
        prompt="p@v1",
        timings=Timings(retrieval_ms=1, generation_ms=1),
        contexts=["ctx"] if contexts is None else contexts,
    )


def _judge(value: float = 0.8, fail: str | None = None) -> tuple[Judge, list[dict[str, Any]]]:
    calls: list[dict[str, Any]] = []

    def make(name: str) -> Any:
        async def metric(q: str, a: str, ctx: list[str], ref: str) -> Any:
            calls.append({"metric": name, "answer": a, "contexts": ctx})
            if name == fail:
                raise ValueError("unparsable judge output")
            return SimpleNamespace(value=value)

        return metric

    return Judge(metrics={m: make(m) for m in JUDGED_METRICS}), calls


async def test_judge_strips_citation_markers_before_scoring() -> None:
    judge, calls = _judge()

    scores, failures = await judge.score("q?", _answer("HNSW is a graph [1][2]."), "ref")

    assert all(v == 0.8 for v in scores.values())
    assert failures == []
    assert calls[0]["answer"] == "HNSW is a graph."


async def test_refusal_scores_zero_for_answer_metrics_and_skips_faithfulness() -> None:
    judge, calls = _judge()

    scores, _ = await judge.score("q?", _answer("I don't know.", refused=True), "ref")

    assert scores["faithfulness"] is None
    assert scores["answer_relevancy"] == 0.0
    assert scores["factual_correctness"] == 0.0
    assert {c["metric"] for c in calls} == {"context_precision", "context_recall"}


async def test_judge_failure_is_recorded_not_raised() -> None:
    judge, _ = _judge(fail="faithfulness")

    scores, failures = await judge.score("q?", _answer("Some answer [1]."), "ref")

    assert scores["faithfulness"] is None
    assert failures == ["faithfulness: ValueError"]


def _row(
    answerable: bool, refused: bool, score: float | None, seconds: float = 1.0
) -> dict[str, Any]:
    return {
        "answerable": answerable,
        "refused": refused,
        "self_check_passed": True,
        "seconds": seconds,
        "scores": {m: score for m in JUDGED_METRICS},
        "judge_failures": [],
    }


def test_summarize_mode_and_markdown_deltas() -> None:
    result = ModeResult(
        "agentic",
        rows=[_row(True, False, 1.0), _row(True, True, None), _row(False, True, None, 3.0)],
    )

    summary = summarize_mode(result)

    assert summary["answered"] == {"value": 0.5, "n": 1, "of": 2}
    assert summary["declined_unanswerable"]["value"] == 1.0
    assert summary["faithfulness"] == {"mean": 1.0, "n": 1}

    report = {
        "run_id": "new",
        "config": {
            "git_sha": "abc",
            "git_dirty": False,
            "dataset_sha": "d",
            "llm": "qwen",
            "retrieval_mode": "hybrid+rerank",
            "judge": "gemma",
            "judge_is_generator": False,
        },
        "retrieval": {"hit_at_1": 0.8, "hit_at_5": 1.0, "mrr_at_10": 0.9, "p50_ms": 300},
        "modes": {"agentic": summary},
    }
    worse = {**summary, "faithfulness": {"mean": 0.9, "n": 1}}
    baseline = {"run_id": "old", "modes": {"agentic": worse}}

    markdown = render_markdown(report, baseline)

    assert "| faithfulness | 1.000 (+0.100) |" in markdown
    assert "Deltas are against `old`." in markdown
