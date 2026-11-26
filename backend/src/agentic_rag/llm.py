"""Chat-model access behind a small protocol, so local and hosted providers are interchangeable."""

from dataclasses import dataclass
from typing import Literal, Protocol

import httpx

Role = Literal["system", "user", "assistant"]


class LLMError(RuntimeError):
    pass


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

    async def chat(self, messages: list[ChatMessage], *, temperature: float = 0.0) -> LLMResult: ...


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

    async def chat(self, messages: list[ChatMessage], *, temperature: float = 0.0) -> LLMResult:
        payload = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "options": {"temperature": temperature, "num_ctx": self._num_ctx},
        }
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
