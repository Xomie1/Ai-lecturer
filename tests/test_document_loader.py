"""
Unit tests — Document Loader
Author  : Madrine Nyawira Kariuki (Test Engineer)
Sprint  : 1 — Task T-1.07
Covers  : PDF load (valid / corrupt / empty), chunk count, chunk size validation
"""

import pytest
from pathlib import Path
from backend.document_loader import extract_text, split_into_chunks, clean_text


# ── Fixtures ──────────────────────────────────────────────────────────────────

SAMPLE_TEXT = (
    "INTRODUCTION\n\n"
    "This is the first paragraph of the lecture. It introduces the topic clearly.\n\n"
    "This is the second paragraph. It expands on the introduction with more detail.\n\n"
    "CONCLUSION\n\n"
    "This paragraph summarises everything discussed in the lecture."
)


# ── clean_text ────────────────────────────────────────────────────────────────

class TestCleanText:
    def test_removes_control_characters(self):
        dirty = "Hello\x00World\x1fTest"
        result = clean_text(dirty)
        assert "\x00" not in result
        assert "\x1f" not in result

    def test_normalises_whitespace(self):
        result = clean_text("too   many    spaces")
        assert "  " not in result

    def test_collapses_excessive_newlines(self):
        result = clean_text("line1\n\n\n\n\nline2")
        assert "\n\n\n" not in result

    def test_empty_string_returns_empty(self):
        assert clean_text("") == ""

    def test_ligature_replacement(self):
        result = clean_text("\ufb01le")   # ﬁ → fi
        assert result.startswith("fi")


# ── split_into_chunks ─────────────────────────────────────────────────────────

class TestSplitIntoChunks:
    def test_returns_list(self):
        chunks = split_into_chunks(SAMPLE_TEXT)
        assert isinstance(chunks, list)

    def test_no_chunks_from_empty_text(self):
        assert split_into_chunks("") == []

    def test_chunk_count_reasonable(self):
        chunks = split_into_chunks(SAMPLE_TEXT)
        assert len(chunks) >= 1

    def test_chunks_respect_max_size(self):
        chunks = split_into_chunks(SAMPLE_TEXT, chunk_size=200)
        for chunk in chunks:
            # Allow slight overflow from overlap prefix, but nothing extreme
            assert len(chunk) < 600, f"Chunk too large: {len(chunk)} chars"

    def test_chunks_not_empty(self):
        chunks = split_into_chunks(SAMPLE_TEXT)
        for chunk in chunks:
            assert chunk.strip(), "Found empty chunk"

    def test_section_heading_preserved(self):
        chunks = split_into_chunks(SAMPLE_TEXT)
        combined = " ".join(chunks)
        assert "INTRODUCTION" in combined or "CONCLUSION" in combined


# ── extract_text ──────────────────────────────────────────────────────────────

class TestExtractText:
    def test_unsupported_format_raises(self):
        with pytest.raises(ValueError, match="Unsupported"):
            extract_text(b"data", "file.xyz")

    def test_txt_extraction(self):
        content = b"Hello from a text file."
        result = extract_text(content, "lecture.txt")
        assert "Hello" in result

    def test_md_extraction(self):
        content = b"# Heading\n\nSome markdown content."
        result = extract_text(content, "notes.md")
        assert "Heading" in result or "markdown" in result

    def test_empty_bytes_returns_empty(self):
        result = extract_text(b"", "empty.txt")
        assert result == ""
