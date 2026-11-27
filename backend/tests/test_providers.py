import json

import httpx
import pytest
from pydantic import BaseModel

from agentic_rag.config import get_settings
from agentic_rag.embeddings import OllamaEmbedder, SentenceTransformerEmbedder
from agentic_rag.llm import (
    ChatMessage,
    LLMError,
    OllamaProvider,
    OpenAICompatibleProvider,
    chat_structured,
)
from agentic_rag.providers import ConfigurationError, build_embedder, build_llm, uses_ollama


def _provider(handler: httpx.MockTransport, retries: int = 2) -> OpenAICompatibleProvider:
    client = httpx.AsyncClient(transport=handler)
    return OpenAICompatibleProvider(
        client, "https://api.example/v1", "secret-key", "m", max_retries=retries
    )


def _completion(text: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": text}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2},
        },
    )


async def test_openai_compatible_chat_sends_auth_and_parses_usage() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers["authorization"]
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return _completion("hello")

    result = await _provider(httpx.MockTransport(handler)).chat([ChatMessage("user", "hi")])

    assert seen["auth"] == "Bearer secret-key"
    assert seen["url"] == "https://api.example/v1/chat/completions"
    assert (result.text, result.prompt_tokens, result.completion_tokens) == ("hello", 10, 2)


class Verdict(BaseModel):
    supported: bool


async def test_structured_output_uses_json_mode_and_states_the_schema() -> None:
    bodies: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return _completion('{"supported": true}')

    verdict = await chat_structured(
        _provider(httpx.MockTransport(handler)),
        [ChatMessage("system", "Judge."), ChatMessage("user", "ok?")],
        Verdict,
    )

    assert verdict.supported
    assert bodies[0]["response_format"] == {"type": "json_object"}
    system = bodies[0]["messages"][0]["content"]  # type: ignore[index]
    assert system.startswith("Judge.")
    assert "supported" in system


async def test_rate_limit_is_retried_then_succeeds() -> None:
    replies = iter([httpx.Response(429, headers={"retry-after": "0"}), _completion("ok")])

    result = await _provider(httpx.MockTransport(lambda _: next(replies))).chat(
        [ChatMessage("user", "hi")]
    )

    assert result.text == "ok"


async def test_errors_never_include_the_api_key() -> None:
    transport = httpx.MockTransport(lambda _: httpx.Response(401, json={"error": "bad key"}))

    with pytest.raises(LLMError) as info:
        await _provider(transport).chat([ChatMessage("user", "hi")])

    assert "401" in str(info.value)
    assert "secret-key" not in str(info.value)


async def test_build_llm_and_embedder_follow_settings() -> None:
    http = httpx.AsyncClient()
    base = get_settings().model_copy(
        update={"llm_provider": "ollama", "embedding_provider": "ollama", "llm_api_key": None}
    )

    assert isinstance(build_llm(base, http), OllamaProvider)
    assert isinstance(build_embedder(base, http), OllamaEmbedder)
    assert uses_ollama(base)

    hosted = base.model_copy(
        update={"llm_provider": "openai_compatible", "embedding_provider": "local"}
    )
    with pytest.raises(ConfigurationError, match="API_KEY"):
        build_llm(hosted, http)
    local = build_embedder(hosted, http)
    assert isinstance(local, SentenceTransformerEmbedder)
    assert local.model == base.embedding_model  # same fingerprint as Ollama: no forced re-index
    assert not uses_ollama(hosted)
