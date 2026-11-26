---
paths:
  - "backend/migrations/**"
  - "backend/src/agentic_rag/db.py"
  - "backend/src/agentic_rag/**/models.py"
  - "docker-compose.yml"
---

# Database rules

- Schema changes go through Alembic only. Never edit a migration that has been applied; add a
  new one. Revision ids are sequential four-digit strings (`0002`, `0003`, ...), and file names
  follow `YYYYMMDD_<rev>_<slug>.py`.
- Every migration has a working `downgrade()`. Verify with
  `uv run alembic downgrade -1 && uv run alembic upgrade head`.
- Vector columns use `vector(<EMBEDDING_DIM>)`. Use HNSW indexes with the operator class that
  matches the distance used in queries (`vector_cosine_ops` for cosine `<=>`).
- Use SQLAlchemy Core/ORM with bound parameters. Never build SQL with f-strings from user input.
- `make reset` deletes the database volume: ask before running it.
