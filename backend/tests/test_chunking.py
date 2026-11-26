from agentic_rag.ingestion.chunking import MIN_CHUNK_CHARS, chunk_document
from agentic_rag.ingestion.loaders import LoadedDocument, Section


def _doc(*sections: Section) -> LoadedDocument:
    return LoadedDocument(
        source="s.md",
        title="Guide",
        media_type="text/markdown",
        content_hash="x",
        sections=list(sections),
    )


def test_chunks_never_cross_section_boundaries() -> None:
    doc = _doc(
        Section("alpha " * 100, heading="A"),
        Section("beta " * 100, heading="B"),
    )

    chunks = chunk_document(doc, chunk_size=200, chunk_overlap=20)

    assert {c.section for c in chunks} == {"A", "B"}
    for c in chunks:
        word = "alpha" if c.section == "A" else "beta"
        assert set(c.content.split()) == {word}


def test_chunk_size_is_respected_and_indexes_are_sequential() -> None:
    doc = _doc(Section("word " * 500, page=3))

    chunks = chunk_document(doc, chunk_size=200, chunk_overlap=20)

    assert len(chunks) > 1
    assert all(len(c.content) <= 200 for c in chunks)
    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert all(c.page == 3 for c in chunks)


def test_tiny_fragments_are_dropped() -> None:
    doc = _doc(Section("x" * (MIN_CHUNK_CHARS - 1)), Section("long enough content " * 5))

    chunks = chunk_document(doc, chunk_size=500, chunk_overlap=0)

    assert len(chunks) == 1


def test_embedding_text_is_prefixed_with_location_but_content_is_not() -> None:
    doc = _doc(Section("Set m to 16 for most workloads, it balances recall.", heading="HNSW"))

    [chunk] = chunk_document(doc, chunk_size=500, chunk_overlap=0)

    assert chunk.embedding_text.startswith("Guide > HNSW\n\n")
    assert chunk.content.startswith("Set m to 16")
