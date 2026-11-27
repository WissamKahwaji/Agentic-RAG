import contextlib
import json
from collections.abc import Iterator
from typing import Any

import pytest

from agentic_rag import tracing
from agentic_rag.agent import AgentAnswerService, AgentConfig
from agentic_rag.config import get_settings
from tests.fakes import FakeLLM, FakeRetriever, make_chunk


class RecordingClient:
    """Stands in for langfuse.Langfuse: records observations, updates and scores."""

    def __init__(self) -> None:
        self.observations: list[dict[str, Any]] = []
        self.scores: dict[str, Any] = {}

    @contextlib.contextmanager
    def start_as_current_observation(
        self, *, name: str, as_type: str, **fields: Any
    ) -> Iterator[Any]:
        record: dict[str, Any] = {"name": name, "as_type": as_type, **fields}
        self.observations.append(record)

        class Obs:
            def update(self, **kw: Any) -> None:
                record.update(kw)

        yield Obs()

    def score_current_trace(self, *, name: str, value: Any, **_: Any) -> None:
        self.scores[name] = value

    def get_trace_url(self) -> str:
        return "http://langfuse.local/trace/1"

    def flush(self) -> None: ...


class AlwaysRelevant:
    async def score(self, query: str, passages: list[str]) -> list[float]:
        return [0.9 for _ in passages]


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> RecordingClient:
    recorder = RecordingClient()
    monkeypatch.setattr(tracing, "_client", recorder)
    return recorder


async def test_answer_is_traced_with_steps_generations_and_scores(client: RecordingClient) -> None:
    def reply(messages: list[Any], schema: Any) -> str:
        return json.dumps({"supported": True}) if schema else "HNSW builds a graph [1]."

    llm = tracing.TracedLLM(FakeLLM(reply))
    service = AgentAnswerService(
        FakeRetriever([make_chunk("HNSW builds a graph.")]), AlwaysRelevant(), llm, 5, AgentConfig()
    )

    answer = await service.answer("What is HNSW?")

    names = [o["name"] for o in client.observations]
    assert names[0] == "answer"
    assert {"retrieve", "grade", "generate", "self_check", "llm"} <= set(names)
    generation = next(o for o in client.observations if o["name"] == "llm")
    assert generation["as_type"] == "generation"
    assert generation["output"] == "HNSW builds a graph [1]."
    assert client.scores == {"grounded": 1.0, "refused": 0.0, "self_check_passed": 1.0}
    assert answer.trace_url == "http://langfuse.local/trace/1"


def test_tracing_is_off_without_keys() -> None:
    settings = get_settings().model_copy(
        update={"langfuse_public_key": None, "langfuse_secret_key": None}
    )

    assert tracing.init_tracing(settings) is False
    assert not tracing.enabled()
    with tracing.observe("x") as obs:
        obs.update(output="ignored")
