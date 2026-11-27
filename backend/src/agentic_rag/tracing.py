"""Optional tracing with Langfuse (OpenTelemetry under the hood).

Enabled only when LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set; otherwise every helper
here is a no-op, so the app, tests and CI never depend on a Langfuse server. Each answered
question becomes one trace: a root observation per question, one per agent step, and one
"generation" per LLM call with its prompt, output, token usage and latency.
"""

import contextlib
import logging
from collections.abc import Iterator
from typing import Any, Literal

from agentic_rag import __version__
from agentic_rag.config import Settings
from agentic_rag.llm import ChatMessage, LLMProvider, LLMResult

logger = logging.getLogger(__name__)

ObservationType = Literal["span", "agent", "chain", "retriever", "evaluator", "generation"]

_client: Any = None  # langfuse.Langfuse when enabled


class _NoopObservation:
    def update(self, **_: Any) -> None:
        return None


def init_tracing(settings: Settings) -> bool:
    """Configure the Langfuse client from settings. Returns whether tracing is on."""
    global _client
    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        _client = None
        return False
    from langfuse import Langfuse  # imported lazily: tracing is optional

    _client = Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key.get_secret_value(),
        host=settings.langfuse_host,
        environment=settings.app_env,
        release=__version__,
    )
    logger.info("tracing to Langfuse at %s", settings.langfuse_host)
    return True


def enabled() -> bool:
    return _client is not None


@contextlib.contextmanager
def observe(name: str, as_type: ObservationType = "span", **fields: Any) -> Iterator[Any]:
    """Open an observation nested under the current one; yields an object with `.update()`."""
    if _client is None:
        yield _NoopObservation()
        return
    with _client.start_as_current_observation(name=name, as_type=as_type, **fields) as obs:
        yield obs


def score_trace(name: str, value: float | str | bool, comment: str | None = None) -> None:
    """Attach a score (e.g. grounded, self-check) to the trace of the current request."""
    if _client is None:
        return
    if isinstance(value, bool):
        _client.score_current_trace(
            name=name, value=1.0 if value else 0.0, data_type="BOOLEAN", comment=comment
        )
    else:
        _client.score_current_trace(name=name, value=value, comment=comment)


def current_trace_url() -> str | None:
    if _client is None:
        return None
    url: str | None = _client.get_trace_url()
    return url


def flush() -> None:
    if _client is not None:
        _client.flush()


class TracedLLM:
    """Wraps any LLMProvider so every call is recorded as a Langfuse generation."""

    def __init__(self, inner: LLMProvider) -> None:
        self._inner = inner
        self.model = inner.model

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.0,
        json_schema: dict[str, Any] | None = None,
    ) -> LLMResult:
        with observe(
            "llm",
            as_type="generation",
            model=self.model,
            input=[{"role": m.role, "content": m.content} for m in messages],
            model_parameters={"temperature": temperature, "json": json_schema is not None},
        ) as generation:
            result = await self._inner.chat(
                messages, temperature=temperature, json_schema=json_schema
            )
            usage = {
                k: v
                for k, v in (
                    ("input", result.prompt_tokens),
                    ("output", result.completion_tokens),
                )
                if v is not None
            }
            generation.update(output=result.text, usage_details=usage or None)
            return result
