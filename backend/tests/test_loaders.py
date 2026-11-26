from pathlib import Path

import pytest

from agentic_rag.ingestion.loaders import UnsupportedFileTypeError, _normalize_pdf_text, load_file

MARKDOWN = """# Vector Search { #vector-search }

Intro paragraph.

## Indexes

HNSW builds a graph.

```python
# not a heading, inside a code fence
```

### Options

Set m and ef_construction.
"""


def test_load_markdown_splits_by_heading_path(tmp_path: Path) -> None:
    path = tmp_path / "guide.md"
    path.write_text(MARKDOWN)

    doc = load_file(path, source="docs/guide.md")

    assert doc.source == "docs/guide.md"
    assert doc.title == "Vector Search"
    assert doc.media_type == "text/markdown"
    assert [s.heading for s in doc.sections] == [
        "Vector Search",
        "Vector Search > Indexes",
        "Vector Search > Indexes > Options",
    ]
    assert "# not a heading" in doc.sections[1].text


def test_load_text_uses_filename_as_title(tmp_path: Path) -> None:
    path = tmp_path / "release_notes.txt"
    path.write_text("Version 2 adds hybrid search.")

    doc = load_file(path)

    assert doc.title == "release notes"
    assert doc.source == "release_notes.txt"
    assert doc.sections[0].page is None


def test_content_hash_changes_with_content(tmp_path: Path) -> None:
    path = tmp_path / "a.txt"
    path.write_text("one")
    first = load_file(path).content_hash

    path.write_text("two")

    assert load_file(path).content_hash != first


def test_empty_file_is_reported_empty(tmp_path: Path) -> None:
    path = tmp_path / "blank.txt"
    path.write_text("   \n")

    assert load_file(path).is_empty


def test_unsupported_suffix_raises(tmp_path: Path) -> None:
    path = tmp_path / "image.png"
    path.write_bytes(b"\x89PNG")

    with pytest.raises(UnsupportedFileTypeError):
        load_file(path)


def test_pdf_text_is_unwrapped_and_dehyphenated() -> None:
    raw = "Retrieval aug-\nmented models  combine\nparametric and\n  non-parametric memory."

    assert _normalize_pdf_text(raw) == (
        "Retrieval augmented models combine parametric and non-parametric memory."
    )
