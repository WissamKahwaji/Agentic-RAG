"""Turn source files into plain-text sections that carry citation metadata (page or heading).

Loaders are synchronous (pypdf is sync); call them via `asyncio.to_thread` from async code.
"""

import hashlib
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from langchain_text_splitters import MarkdownHeaderTextSplitter
from pypdf import PageObject, PdfReader

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
    """One section per (page, heading) span, so PDF chunks carry heading context like Markdown.

    A section that continues onto the next page keeps its heading there.
    """
    reader = PdfReader(path)
    sections: list[Section] = []
    heading: str | None = None
    for number, page in enumerate(reader.pages, start=1):
        headings: list[str] = []
        raw = page.extract_text(visitor_text=_heading_collector(headings)) or ""
        text = _normalize_pdf_text(raw)
        for part, part_heading in _split_at_headings(text, headings, heading):
            if part.strip():
                sections.append(Section(text=part.strip(), page=number, heading=part_heading))
            heading = part_heading
    meta_title = (reader.metadata.title or "").strip() if reader.metadata else ""
    title = meta_title or (_largest_text(reader.pages[0]) if reader.pages else None)
    return title, sections


# Numbered or lettered headings ("2.3 Generator: BART", "A Implementation Details") and the
# standard unnumbered ones. Bold run-in labels like "RAG-Token Model" are deliberately ignored.
_NUMBERED_HEADING = re.compile(r"^(?:\d{1,2}(?:\.\d{1,2}){0,3}|[A-H])\s+[A-Z][^\n]{1,80}$")
_NAMED_HEADINGS = {
    "Abstract",
    "Acknowledgments",
    "Acknowledgements",
    "Appendix",
    "Appendices",
    "Broader Impact",
    "Conclusion",
    "Conclusions",
    "References",
}
_HEADING_FONT_HINTS = ("bold", "medi", "semibold", "black", "heavy", "cmbx")


def _heading_collector(found: list[str]) -> Callable[..., None]:
    """pypdf text visitor that records numbered headings set in a bold/medium font."""

    def visit(text: str, cm: object, tm: object, font: object, size: float) -> None:
        candidate = " ".join(text.split())
        name = str(font.get("/BaseFont", "")) if isinstance(font, dict) else ""
        is_heading_font = any(hint in name.lower() for hint in _HEADING_FONT_HINTS)
        if is_heading_font and (_NUMBERED_HEADING.match(candidate) or candidate in _NAMED_HEADINGS):
            found.append(candidate)

    return visit


def _split_at_headings(
    text: str, headings: list[str], current: str | None
) -> list[tuple[str, str | None]]:
    """Cut page text where each detected heading occurs; text before the first heading keeps
    the heading carried over from the previous page."""
    parts: list[tuple[str, str | None]] = []
    position = 0
    for heading in headings:
        index = text.find(heading, position)
        if index < 0:
            continue  # extraction joined or reordered the heading; keep the text together
        parts.append((text[position:index], current))
        current = heading
        position = index + len(heading)
    parts.append((text[position:], current))
    return parts


def _largest_text(page: PageObject) -> str | None:
    """Best-effort title: the largest horizontal text on the page (PDF metadata is often empty)."""
    pieces: list[tuple[float, str]] = []

    def visit(text: str, cm: list[float], tm: list[float], font: object, size: float) -> None:
        # tm is the text matrix; skip rotated text such as arXiv's vertical margin stamp.
        if text.strip() and tm[1] == 0 and tm[2] == 0:
            pieces.append((round(size * tm[3], 1), text))

    page.extract_text(visitor_text=visit)
    if not pieces:
        return None
    largest = max(size for size, _ in pieces)
    title = " ".join(" ".join(t for s, t in pieces if s == largest).split())
    return title if 3 <= len(title) <= 200 else None


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


def _normalize_pdf_text(text: str) -> str:
    """Undo the PDF's visual line wrapping so sentences are contiguous.

    PDF text arrives as lines as they were *laid out*, not as written: a sentence is broken at
    every column edge and long words are hyphenated. Left as-is, chunk boundaries and embeddings
    see fragments, and passages render with ragged breaks.
    """
    text = re.sub(r"([a-z])-\n([a-z])", r"\1\2", text)  # re-join hyphenated words
    text = re.sub(r"[ \t]*\n[ \t]*", " ", text)  # wrapped lines -> one flowing paragraph
    return re.sub(r"[ \t]{2,}", " ", text).strip()
