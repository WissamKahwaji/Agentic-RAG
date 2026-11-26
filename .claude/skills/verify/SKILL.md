---
name: verify
description: Run the full quality gate (lint, types, unit and integration tests, live health check) and report results. Use before saying any change is done, before committing, or when asked "does it work?".
---

# Verify

Run from the repo root, in order, and stop at the first failure to fix it:

1. `make format` then `make lint`
2. `make typecheck`
3. `make test`
4. If the change touches the database, retrieval, embeddings or model calls:
   - `docker compose ps` shows postgres healthy (else `make up`)
   - `curl -s localhost:11434/api/tags` responds (else Ollama is not running)
   - `make migrate`
   - `make test-all`
5. If the change touches the API surface, start `make dev` in the background, call the
   affected endpoints with `curl`, then stop the server.

Report as a short table: step, result, and the exact error output for anything that failed.
Do not describe a step as passing unless you ran it in this session.
