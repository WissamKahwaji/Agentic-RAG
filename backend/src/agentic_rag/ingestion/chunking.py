"""Split document sections into overlapping, retrievable chunks.

Chunks never cross a section boundary, so every chunk has exactly one citation location
(a PDF page or a Markdown heading path).
"""

from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from agentic_rag.ingestion.loaders import LoadedDocument

# Below this many characters a chunk is almost always a stray heading or page artefact.
MIN_CHUNK_CHARS = 40


@dataclass(frozen=True)
class ChunkDraft:
    index: int
    content: str
    page: int | None
    section: str | None
    embedding_text: str


def chunk_document(doc: LoadedDocument, chunk_size: int, chunk_overlap: int) -> list[ChunkDraft]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    drafts: list[ChunkDraft] = []
    for section in doc.sections:
        for piece in splitter.split_text(section.text):
            content = piece.strip()
            if len(content) < MIN_CHUNK_CHARS:
                continue
            drafts.append(
                ChunkDraft(
                    index=len(drafts),
                    content=content,
                    page=section.page,
                    section=section.heading,
                    embedding_text=_embedding_text(doc.title, section.heading, content),
                )
            )
    return drafts


def _embedding_text(title: str, heading: str | None, content: str) -> str:
    """Prefix the chunk with where it comes from before embedding.

    A chunk like "Set it to 16 for most workloads." is meaningless alone; with
    "pgvector > HNSW > Index options" in front, its vector lands near questions about HNSW.
    The stored `content` stays unprefixed so citations show the original text.
    """
    location = f"{title} > {heading}" if heading else title
    return f"{location}\n\n{content}"
