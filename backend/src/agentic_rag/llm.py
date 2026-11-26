"""Chat-model access behind a small protocol, so local and hosted providers are interchangeable."""

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
    ) -> None:
        self._http = http
        self._url = f"{base_url.rstrip('/')}/api/chat"
        self.model = model
        # Ollama's default context window is small and it truncates the *start* of the prompt
        # silently when exceeded, which would drop the instructions and the first sources.
        self._num_ctx = num_ctx

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
            "options": {"temperature": temperature, "num_ctx": self._num_ctx},
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
