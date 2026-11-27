"""Chat-model access behind a small protocol, so local and hosted providers are interchangeable."""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import httpx
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

Role = Literal["system", "user", "assistant"]


class LLMError(RuntimeError):
    pass


class StructuredOutputError(LLMError):
    """The model did not produce output matching the requested schema, even after retrying."""


@dataclass(frozen=True)
class ChatMessage:
    role: Role
    content: str


@dataclass(frozen=True)
class LLMResult:
    text: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class LLMProvider(Protocol):
    model: str

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.0,
        json_schema: dict[str, Any] | None = None,
    ) -> LLMResult:
        """Generate a reply. With `json_schema`, constrain the reply to JSON matching it."""
        ...


async def chat_structured[T: BaseModel](
    llm: LLMProvider,
    messages: list[ChatMessage],
    schema: type[T],
    *,
    temperature: float = 0.0,
    retries: int = 1,
) -> T:
    """Ask for JSON matching `schema` and parse it, retrying with the validation error.

    Schema-constrained decoding makes malformed JSON rare, but values can still violate the
    schema's rules, so the parse can fail; one corrective retry fixes most of those.
    """
    attempt_messages = list(messages)
    for attempt in range(retries + 1):
        result = await llm.chat(
            attempt_messages, temperature=temperature, json_schema=schema.model_json_schema()
        )
        try:
            return schema.model_validate_json(result.text)
        except ValidationError as exc:
            logger.warning("structured output attempt %d failed: %s", attempt + 1, exc)
            attempt_messages = [
                *messages,
                ChatMessage("assistant", result.text),
                ChatMessage("user", f"That JSON was invalid: {exc}. Reply again with valid JSON."),
            ]
    raise StructuredOutputError(f"no valid {schema.__name__} after {retries + 1} attempts")


class OllamaProvider:
    """Non-streaming chat through Ollama's `/api/chat`."""

    def __init__(
        self,
        http: httpx.AsyncClient,
        base_url: str,
        model: str,
        num_ctx: int,
        max_tokens: int = 1536,
    ) -> None:
        self._http = http
        self._url = f"{base_url.rstrip('/')}/api/chat"
        self.model = model
        # Ollama's default context window is small and it truncates the *start* of the prompt
        # silently when exceeded, which would drop the instructions and the first sources.
        self._num_ctx = num_ctx
        # Hard cap on generated tokens. Without it a misbehaving generation (common in JSON mode)
        # runs until the context window is full; the timeout alone would turn that into an error
        # only after a long stall.
        self._max_tokens = max_tokens

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.0,
        json_schema: dict[str, Any] | None = None,
    ) -> LLMResult:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_ctx": self._num_ctx,
                "num_predict": self._max_tokens,
            },
        }
        if json_schema is not None:
            payload["format"] = json_schema  # Ollama constrains decoding to this schema
        try:
            resp = await self._http.post(self._url, json=payload)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMError(f"Ollama chat request failed: {exc}") from exc

        body = resp.json()
        text = body.get("message", {}).get("content")
        if not isinstance(text, str):
            raise LLMError("Ollama chat response had no message content")
        return LLMResult(
            text=text,
            model=self.model,
            prompt_tokens=body.get("prompt_eval_count"),
            completion_tokens=body.get("eval_count"),
        )


class OpenAICompatibleProvider:
    """Any OpenAI-compatible chat API: Groq, Gemini's OpenAI endpoint, OpenRouter, vLLM, ...

    JSON output uses `json_object` mode, which every compatible API supports, with the schema
    stated in the system message; `chat_structured` then validates it. Rate limits (429) are
    retried with backoff, honouring Retry-After, because free tiers hit them routinely.
    """

    def __init__(
        self,
        http: httpx.AsyncClient,
        base_url: str,
        api_key: str,
        model: str,
        max_tokens: int = 1536,
        max_retries: int = 3,
    ) -> None:
        self._http = http
        self._url = f"{base_url.rstrip('/')}/chat/completions"
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self.model = model
        self._max_tokens = max_tokens
        self._max_retries = max_retries

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.0,
        json_schema: dict[str, Any] | None = None,
    ) -> LLMResult:
        wire = [{"role": m.role, "content": m.content} for m in messages]
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": wire,
            "temperature": temperature,
            "max_tokens": self._max_tokens,
        }
        if json_schema is not None:
            payload["response_format"] = {"type": "json_object"}
            schema_note = f"\n\nReply with a JSON object matching this JSON Schema:\n{json_schema}"
            if wire and wire[0]["role"] == "system":
                wire[0] = {"role": "system", "content": wire[0]["content"] + schema_note}
            else:
                wire.insert(0, {"role": "system", "content": schema_note.strip()})

        body = await self._post(payload)
        choice = (body.get("choices") or [{}])[0]
        text = choice.get("message", {}).get("content")
        if not isinstance(text, str):
            raise LLMError("chat completion had no message content")
        usage = body.get("usage") or {}
        return LLMResult(
            text=text,
            model=self.model,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
        )

    async def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        for attempt in range(self._max_retries + 1):
            try:
                resp = await self._http.post(self._url, json=payload, headers=self._headers)
            except httpx.HTTPError as exc:
                raise LLMError(f"chat completion request failed: {exc}") from exc
            if resp.status_code == 429 and attempt < self._max_retries:
                delay = _retry_after_seconds(resp, default=2.0 * (attempt + 1))
                logger.warning("rate limited by %s; retrying in %.1fs", self._url, delay)
                await asyncio.sleep(delay)
                continue
            if resp.status_code >= 400:
                # Never log the request (it carries the API key in a header); the body is safe.
                raise LLMError(f"chat completion failed: HTTP {resp.status_code} {resp.text[:300]}")
            result: dict[str, Any] = resp.json()
            return result
        raise LLMError("chat completion failed: still rate limited after retries")


def _retry_after_seconds(resp: httpx.Response, default: float) -> float:
    try:
        return min(float(resp.headers.get("retry-after", default)), 60.0)
    except ValueError:
        return default
