---
paths:
  - "backend/tests/**"
---

# Testing conventions

- pytest with `asyncio_mode = auto`: write `async def test_...` directly, no decorator.
- Unit tests must run with no Docker, no Ollama, no network. Swap dependencies through
  `app.dependency_overrides` or by passing fakes into constructors; avoid `monkeypatch` on
  module internals.
- Anything that needs the live stack gets `@pytest.mark.integration`. It runs in
  `make test-all`, not `make test`.
- LLM output is non-deterministic: unit tests assert on structure (citations parsed, fields
  present, prompt contains the retrieved chunks), never on exact generated wording. Answer
  *quality* belongs in the eval suite, not in unit tests.
- Name tests for behaviour: `test_<unit>_<condition>_<expected>`. Arrange / act / assert,
  separated by blank lines.
- A bug fix starts with a failing test that reproduces it.
