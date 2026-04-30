"""Document loading, text extraction, and smart chunking.

Supported formats: PDF, DOCX, TXT, MD
"""

from __future__ import annotations

import io
import re
from pathlib import Path
from typing import List

import nltk

# Ensure NLTK sentence tokeniser is available
for _resource in ("tokenizers/punkt_tab", "tokenizers/punkt"):
    try:
        nltk.data.find(_resource)
    except LookupError:
        nltk.download(_resource.split("/")[-1], quiet=True)

from nltk.tokenize import sent_tokenize


# ---------------------------------------------------------------------------
# Ligature / character normalisation map (common PDF extraction artefacts)
# ---------------------------------------------------------------------------
_LIGATURE_MAP: dict[str, str] = {
    "\ufb00": "ff",  "\ufb01": "fi",  "\ufb02": "fl",
    "\ufb03": "ffi", "\ufb04": "ffl", "\ufb05": "st", "\ufb06": "st",
    "\uf000": "",    "\uf001": "fi",  "\uf002": "fl",
}


# ---------------------------------------------------------------------------
# Text cleaning helpers
# ---------------------------------------------------------------------------

def _collapse_spaced_letters(text: str) -> str:
    """Collapse 'H e l l o' style spaced letters back into words."""
    pattern = re.compile(
        r"(?<![A-Za-z\d])([A-Za-z])(?:[ \t]([A-Za-z])){2,}(?![A-Za-z\d])"
    )
    def _join(m: re.Match) -> str:
        return m.group(0).replace(" ", "").replace("\t", "")
    for _ in range(3):
        new = pattern.sub(_join, text)
        if new == text:
            break
        text = new
    return text


def _fix_punctuation(text: str) -> str:
    text = re.sub(r"([.!?])([A-Z])",    r"\1 \2", text)
    text = re.sub(r"([,:;])([A-Za-z])", r"\1 \2", text)
    text = re.sub(r"\s*[·•∙]\s*",       " · ",    text)
    text = re.sub(r"(\w)-\n(\w)",        r"\1\2",  text)
    text = re.sub(r"\s*--+\s*",         " — ",    text)
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    return text


def clean_text(text: str) -> str:
    """Normalise extracted text: fix ligatures, spacing, punctuation."""
    if not text:
        return ""
    for lig, rep in _LIGATURE_MAP.items():
        text = text.replace(lig, rep)
    text = _collapse_spaced_letters(text)
    text = _fix_punctuation(text)
    # Strip non-printable control characters
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r"[ \t]+",  " ",   text)
    text = re.sub(r"\n{3,}",  "\n\n", text)
    text = re.sub(r" +\n",    "\n",   text)
    return text.strip()


# ---------------------------------------------------------------------------
# Extraction — per format
# ---------------------------------------------------------------------------

def _extract_pdf(content: bytes) -> str:
    try:
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(content))
        parts = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            if page_text.strip():
                parts.append(page_text.strip())
        return "\n\n".join(parts)
    except ImportError:
        raise ImportError("pypdf is required for PDF support: pip install pypdf")


def _extract_docx(content: bytes) -> str:
    try:
        from docx import Document
    except ImportError:
        raise ImportError("python-docx is required for DOCX support: pip install python-docx")

    doc = Document(io.BytesIO(content))
    parts: list[str] = []
    for block in doc.element.body:
        tag = block.tag.split("}")[-1]
        if tag == "p":
            text = "".join(
                n.text for n in block.iter()
                if n.tag.endswith("}t") and n.text
            )
            if text.strip():
                parts.append(text.strip())
        elif tag == "tbl":
            for row in block.iter():
                if row.tag.endswith("}tr"):
                    cells = []
                    for cell in row.iter():
                        if cell.tag.endswith("}tc"):
                            ct = "".join(
                                n.text for n in cell.iter()
                                if n.tag.endswith("}t") and n.text
                            )
                            if ct.strip():
                                cells.append(ct.strip())
                    if cells:
                        parts.append(" | ".join(cells))
    return "\n\n".join(parts)


def extract_text(content: bytes, filename: str) -> str:
    """Dispatch to the correct extractor and clean the result."""
    name = filename.lower()
    if name.endswith(".pdf"):
        raw = _extract_pdf(content)
    elif name.endswith(".docx"):
        raw = _extract_docx(content)
    elif name.endswith((".txt", ".md")):
        raw = content.decode("utf-8", errors="ignore")
    else:
        raise ValueError(f"Unsupported file type: {filename}")

    result = clean_text(raw)
    print(f"[loader] '{filename}': extracted {len(result)} chars")
    return result


# ---------------------------------------------------------------------------
# Smart chunking
# ---------------------------------------------------------------------------

def _split_by_sections(
    paragraphs: list[str],
) -> list[tuple[str, list[str]]]:
    """Group paragraphs under their nearest UPPER-CASE heading."""
    sections: list[tuple[str, list[str]]] = []
    heading, body = "", []
    for para in paragraphs:
        if para.isupper() and 1 <= len(para.split()) <= 8:
            if body or heading:
                sections.append((heading, body))
            heading, body = para, []
        else:
            body.append(para)
    if body or heading:
        sections.append((heading, body))
    return sections


def split_into_chunks(
    text: str,
    chunk_size: int = 800,
    overlap: int = 100,
) -> list[str]:
    """Split text into overlapping sentence-aware chunks.

    Respects ALL-CAPS section headings found in the document and keeps
    each heading as a prefix on the chunks that follow it, so the LLM
    always knows which section a chunk belongs to.
    """
    if not text:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    sections   = _split_by_sections(paragraphs)
    chunks: list[str] = []

    for heading, body in sections:
        section_text = (heading + "\n\n" if heading else "") + "\n\n".join(body)
        section_text = section_text.strip()
        if not section_text:
            continue

        if len(section_text) <= chunk_size:
            chunks.append(section_text)
        else:
            prefix    = (heading + "\n\n") if heading else ""
            sentences = sent_tokenize(section_text)
            temp      = prefix
            for sent in sentences:
                if len(temp) + len(sent) + 1 <= chunk_size:
                    temp += (" " if temp else "") + sent
                else:
                    if temp.strip():
                        chunks.append(temp.strip())
                    temp = prefix + sent
            if temp.strip():
                chunks.append(temp.strip())

    # Add trailing overlap so adjacent chunks share context
    if len(chunks) > 1 and overlap > 0:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            overlapped.append(chunks[i - 1][-overlap:] + "\n\n" + chunks[i])
        return overlapped

    return chunks


# ---------------------------------------------------------------------------
# Public convenience: load a file from disk
# ---------------------------------------------------------------------------

def load_file(path: Path) -> tuple[str, list[str]]:
    """Read a file and return (raw_text, chunks).

    Returns an empty string / empty list on failure.
    """
    try:
        content = path.read_bytes()
        text    = extract_text(content, path.name)
        chunks  = split_into_chunks(text)
        return text, chunks
    except Exception as exc:
        print(f"[loader] Failed to load '{path.name}': {exc}")
        return "", []
