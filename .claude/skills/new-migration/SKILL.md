---
name: new-migration
description: Create, apply and round-trip test a new Alembic database migration. Use whenever a table, column, index or extension needs to change.
---

# New migration

1. Find the latest revision: `ls backend/migrations/versions/` (ids are `0001`, `0002`, ...).
2. Create `backend/migrations/versions/<YYYYMMDD>_<next-id>_<slug>.py` by copying the header
   of the previous file. Set `revision = "<next-id>"` and `down_revision` to the previous id.
3. Write `upgrade()` and a real `downgrade()` that fully reverses it. Use `op.create_table`,
   `op.add_column`, etc.; use `op.execute` only for things Alembic cannot express (extensions,
   HNSW indexes, generated tsvector columns).
4. Apply and round-trip:
   ```bash
   cd backend
   uv run alembic upgrade head
   uv run alembic downgrade -1
   uv run alembic upgrade head
   ```
5. If ORM models changed, update them in the same change so code and schema never diverge.
6. Run the `verify` skill.
