import httpx

from agentic_rag.web import MetaWebSearch, TavilyWebSearch, WebResult


async def test_meta_search_retries_once_after_an_empty_round(monkeypatch: object) -> None:
    search = MetaWebSearch(["a"], retry_after_s=0.0001)
    rounds = iter([[], [WebResult("t", "https://x.dev", "body")]])
    search._search_sync = lambda q, k: next(rounds)  # type: ignore[method-assign]

    results = await search.search("q", 3)

    assert [r.url for r in results] == ["https://x.dev"]


async def test_tavily_maps_results_and_swallows_http_errors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer key"
        return httpx.Response(
            200,
            json={"results": [{"title": "T", "url": "https://a.dev", "content": "C"}, {"url": ""}]},
        )

    ok = TavilyWebSearch(httpx.AsyncClient(transport=httpx.MockTransport(handler)), "key")
    down = TavilyWebSearch(
        httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(429))), "key"
    )

    assert await ok.search("q", 5) == [WebResult("T", "https://a.dev", "C")]
    assert await down.search("q", 5) == []
