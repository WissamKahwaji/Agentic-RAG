"""Add a generated tsvector column with a GIN index for keyword (full-text) search.

Revision ID: 0003
Revises: 0002
Create Date: 2026-11-26
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Generated column: Postgres keeps it in sync with section/content on every write, so the
    # ingestion code never has to remember to maintain it. Heading words weigh more (A) than
    # body words (B) because a match in "HNSW > Query Options" is stronger evidence.
    op.execute(
        """
        ALTER TABLE chunks ADD COLUMN search_vector tsvector
        GENERATED ALWAYS AS (
            setweight(to_tsvector('english', coalesce(section, '')), 'A') ||
            setweight(to_tsvector('english', content), 'B')
        ) STORED
        """
    )
    op.create_index("ix_chunks_search_vector", "chunks", ["search_vector"], postgresql_using="gin")


def downgrade() -> None:
    op.drop_index("ix_chunks_search_vector", table_name="chunks")
    op.drop_column("chunks", "search_vector")
