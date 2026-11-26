"""Turn source files into plain-text sections that carry citation metadata (page or heading).

Loaders are synchronous (pypdf is sync); call them via `asyncio.to_thread` from async code.
"""

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

from langchain_text_splitters import MarkdownHeaderTextSplitter
from pypdf import PdfReader

MARKDOWN_SUFFIXES = {".md", ".markdown"}
TEXT_SUFFIXES = {".txt"}
PDF_SUFFIXES = {".pdf"}
SUPPORTED_SUFFIXES = MARKDOWN_SUFFIXES | TEXT_SUFFIXES | PDF_SUFFIXES

_MD_HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]
_FIRST_H1 = re.compile(r"^#\s+(.+?)\s*#*\s*$", re.MULTILINE)
# Heading attribute blocks such as `## First Steps { #first-steps }` (MkDocs anchors).
_HEADING_ATTRS = re.compile(r"\s*\{\s*#[^}]*\}\s*$")


class UnsupportedFileTypeError(ValueError):
    pass


@dataclass(frozen=True)
class Section:
    """A contiguous span of a document that shares one citation location."""

    text: str
    page: int | None = None
    heading: str | None = None


@dataclass(frozen=True)
class LoadedDocument:
    source: str
    title: str
    media_type: str
    content_hash: str
    sections: list[Section] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not any(s.text.strip() for s in self.sections)


def is_supported(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_SUFFIXES


def load_file(path: Path, source: str | None = None) -> LoadedDocument:
    """Load one file. `source` is the stable identifier stored for citations (default: name)."""
    suffix = path.suffix.lower()
    raw = path.read_bytes()
    source = source or path.name
    content_hash = hashlib.sha256(raw).hexdigest()

    if suffix in PDF_SUFFIXES:
        title, sections = _load_pdf(path)
        media_type = "application/pdf"
    elif suffix in MARKDOWN_SUFFIXES:
        title, sections = _load_markdown(raw.decode("utf-8", errors="replace"))
        media_type = "text/markdown"
    elif suffix in TEXT_SUFFIXES:
        title, sections = None, [Section(text=raw.decode("utf-8", errors="replace"))]
        media_type = "text/plain"
    else:
        raise UnsupportedFileTypeError(f"unsupported file type: {path.suffix or path.name}")

    return LoadedDocument(
        source=source,
        title=title or path.stem.replace("-", " ").replace("_", " "),
        media_type=media_type,
        content_hash=content_hash,
        sections=sections,
    )


def _load_pdf(path: Path) -> tuple[str | None, list[Section]]:
    reader = PdfReader(path)
    sections = [
        Section(text=text, page=number)
        for number, page in enumerate(reader.pages, start=1)
        if (text := _normalize_whitespace(page.extract_text() or ""))
    ]
    meta_title = reader.metadata.title if reader.metadata else None
    return (meta_title.strip() if meta_title and meta_title.strip() else None), sections


def _load_markdown(text: str) -> tuple[str | None, list[Section]]:
    splitter = MarkdownHeaderTextSplitter(_MD_HEADERS, strip_headers=False)
    sections = []
    for part in splitter.split_text(text):
        heading = " > ".join(
            _clean_heading(str(part.metadata[k])) for _, k in _MD_HEADERS if k in part.metadata
        )
        sections.append(Section(text=part.page_content, heading=heading or None))
    match = _FIRST_H1.search(text)
    return (_clean_heading(match.group(1)) if match else None), sections


def _clean_heading(text: str) -> str:
    return _HEADING_ATTRS.sub("", text).strip()


def _normalize_whitespace(text: str) -> str:
    # PDF extraction yields ragged spacing; collapse runs but keep paragraph breaks.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
