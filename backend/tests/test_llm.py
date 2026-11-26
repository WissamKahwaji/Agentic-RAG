import json

import httpx
import pytest

from agentic_rag.llm import ChatMessage, LLMError, OllamaProvider


def _provider(handler: httpx.MockTransport) -> OllamaProvider:
    return OllamaProvider(httpx.AsyncClient(transport=handler), "http://ollama", "qwen", 8192)


async def test_chat_sends_context_window_and_returns_text_and_usage() -> None:
    sent: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        sent.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={"message": {"content": "hi"}, "prompt_eval_count": 12, "eval_count": 3},
        )

    result = await _provider(httpx.MockTransport(handler)).chat(
        [ChatMessage("user", "hello")], temperature=0.2
    )

    assert sent["options"] == {"temperature": 0.2, "num_ctx": 8192}
    assert sent["stream"] is False
    assert (result.text, result.prompt_tokens, result.completion_tokens) == ("hi", 12, 3)


async def test_chat_wraps_http_errors() -> None:
    transport = httpx.MockTransport(lambda _: httpx.Response(500))

    with pytest.raises(LLMError):
        await _provider(transport).chat([ChatMessage("user", "hello")])
