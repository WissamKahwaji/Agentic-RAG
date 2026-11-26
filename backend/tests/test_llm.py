import json

import httpx
import pytest
from pydantic import BaseModel

from agentic_rag.llm import (
    ChatMessage,
    LLMError,
    OllamaProvider,
    StructuredOutputError,
    chat_structured,
)
from tests.fakes import FakeLLM


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

    assert sent["options"] == {"temperature": 0.2, "num_ctx": 8192, "num_predict": 1536}
    assert sent["stream"] is False
    assert (result.text, result.prompt_tokens, result.completion_tokens) == ("hi", 12, 3)


async def test_chat_wraps_http_errors() -> None:
    transport = httpx.MockTransport(lambda _: httpx.Response(500))

    with pytest.raises(LLMError):
        await _provider(transport).chat([ChatMessage("user", "hello")])


class Verdict(BaseModel):
    supported: bool


async def test_chat_structured_sends_schema_and_parses() -> None:
    llm = FakeLLM('{"supported": true}')

    verdict = await chat_structured(llm, [ChatMessage("user", "ok?")], Verdict)

    assert verdict.supported is True
    assert llm.schemas[0] == Verdict.model_json_schema()


async def test_chat_structured_retries_once_with_the_error() -> None:
    replies = iter(['{"supported": "maybe"}', '{"supported": false}'])
    llm = FakeLLM(lambda _messages, _schema: next(replies))

    verdict = await chat_structured(llm, [ChatMessage("user", "ok?")], Verdict)

    assert verdict.supported is False
    assert "invalid" in llm.received[1][-1].content


async def test_chat_structured_gives_up_after_retries() -> None:
    with pytest.raises(StructuredOutputError):
        await chat_structured(FakeLLM("not json"), [ChatMessage("user", "ok?")], Verdict)
