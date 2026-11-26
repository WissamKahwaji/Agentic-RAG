---
name: local-stack
description: Start, check or troubleshoot the local development stack (Colima/Docker, Postgres + pgvector, Ollama, the API). Use when something "can't connect", after a reboot, or before integration work.
---

# Local stack

Check each layer bottom-up and fix the first broken one:

| Layer    | Check                                   | Fix                                         |
| -------- | --------------------------------------- | ------------------------------------------- |
| Docker   | `docker info >/dev/null && echo ok`     | `colima start` (or `colima status` to inspect) |
| Postgres | `docker compose ps` shows `healthy`     | `make up`; logs: `docker compose logs postgres` |
| Schema   | `cd backend && uv run alembic current`  | `make migrate`                              |
| Ollama   | `curl -s localhost:11434/api/tags`      | `brew services start ollama`                |
| Models   | `ollama list` shows `qwen2.5:7b`, `bge-m3` | `ollama pull <model>`                    |
| API      | `curl -s localhost:8000/health`         | `make dev`; every check in the JSON must be `ok` |

Notes:
- Ollama runs on the host, not in Docker, so the API reaches it at `localhost:11434`.
- Port 5432 already in use: another Postgres is running. Set `POSTGRES_PORT` and the port in
  `DATABASE_URL` in `.env` to a free port instead of stopping someone else's database.
- Never run `make reset` (deletes all data) without asking first.
