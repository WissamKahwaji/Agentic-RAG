"""Web search behind a small protocol.

The default needs no account: the `ddgs` metasearch library, trying several backends in order
because keyless scraping endpoints are rate-limited or blocked unpredictably. A keyed API
(Tavily) is the production choice and is used when TAVILY_API_KEY is set.

Web content is untrusted: results are only ever passed to the model as numbered *data*, and the
grounded-answer prompt tells it to ignore instructions inside sources.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Protocol

import httpx

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WebResult:
    title: str
    url: str
    content: str


class WebSearch(Protocol):
    name: str

    async def search(self, query: str, k: int) -> list[WebResult]:
        """Up to `k` results; an empty list when search is unavailable (never raises)."""
        ...


class MetaWebSearch:
    """Keyless search through `ddgs`, falling through backends until one returns results."""

    name = "ddgs"

    def __init__(
        self, backends: list[str], timeout_s: float = 10.0, retry_after_s: float = 2.0
    ) -> None:
        self._backends = backends
        self._timeout_s = timeout_s
        self._retry_after_s = retry_after_s

    async def search(self, query: str, k: int) -> list[WebResult]:
        # Keyless endpoints rate-limit bursts; one spaced retry recovers most failures
        # (measured: 5/6 queries succeed when spaced, far fewer in a burst).
        results = await asyncio.to_thread(self._search_sync, query, k)
        if not results and self._retry_after_s:
            await asyncio.sleep(self._retry_after_s)
            results = await asyncio.to_thread(self._search_sync, query, k)
        if not results:
            logger.warning("web search returned nothing from any backend for %r", query)
        return results

    def _search_sync(self, query: str, k: int) -> list[WebResult]:
        from ddgs import DDGS  # imported lazily: only needed in hybrid mode

        for backend in self._backends:
            try:
                rows = DDGS(timeout=int(self._timeout_s)).text(
                    query, max_results=k, backend=backend
                )
            except Exception as exc:  # the library raises on "no results" and on blocking
                logger.info("web search backend %s failed: %s", backend, exc)
                continue
            results = [
                WebResult(
                    title=r.get("title", ""), url=r.get("href", ""), content=r.get("body", "")
                )
                for r in rows
                if r.get("href") and r.get("body")
            ]
            if results:
                return results[:k]
        return []


class TavilyWebSearch:
    """Tavily's search API (free tier available); returns page extracts, not just snippets."""

    name = "tavily"

    def __init__(self, http: httpx.AsyncClient, api_key: str) -> None:
        self._http = http
        self._api_key = api_key

    async def search(self, query: str, k: int) -> list[WebResult]:
        try:
            resp = await self._http.post(
                "https://api.tavily.com/search",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={"query": query, "max_results": k, "search_depth": "basic"},
                timeout=20.0,
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning("tavily search failed: %s", exc)
            return []
        return [
            WebResult(title=r.get("title", ""), url=r["url"], content=r.get("content", ""))
            for r in resp.json().get("results", [])
            if r.get("url") and r.get("content")
        ]
