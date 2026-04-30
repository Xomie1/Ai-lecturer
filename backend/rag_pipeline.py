"""RAG pipeline — retrieval + context assembly + answer generation.

Retrieval strategy
------------------
* Broad queries (list / summarise / how many / etc.) → fetch ALL indexed chunks
* Focused queries → Hybrid BM25 + vector search fused with RRF

The two-path approach means summary questions get the full picture while
specific questions get the most targeted context, which matters a lot for
a 3B parameter model on low-spec hardware.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Generator, List, Tuple

from .document_loader import extract_text, split_into_chunks
from .vector_store import VectorStore
from .llama_model import LlamaModel

# ---------------------------------------------------------------------------
# Broad-query detection
# ---------------------------------------------------------------------------
_BROAD_RE = re.compile(
    r"\b(list|all|every|each|full list|how many|count|total|complete|"
    r"summaris|summariz|overview|tell me about everyone|who are)\b",
    re.IGNORECASE,
)


def _is_broad_query(msg: str) -> bool:
    return bool(_BROAD_RE.search(msg))


# ---------------------------------------------------------------------------
# RagPipeline
# ---------------------------------------------------------------------------

class RagPipeline:
    def __init__(
        self,
        documents_path: Path,
        vector_db_path: Path,
        llm_model: str = "qwen2.5:3b",
    ) -> None:
        self._documents_path = documents_path
        self._store = VectorStore(vector_db_path)
        self._llm   = LlamaModel(model_name=llm_model)

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest_bytes(self, content: bytes, filename: str) -> int:
        """Extract, chunk, and index a file supplied as raw bytes.

        Returns the number of chunks added to the index.
        """
        text = extract_text(content, filename)
        if not text or len(text) < 50:
            raise ValueError(f"Could not extract usable text from '{filename}'")
        chunks = split_into_chunks(text)
        if not chunks:
            raise ValueError(f"No chunks produced from '{filename}'")
        added = self._store.add_chunks(chunks, filename)
        print(f"[rag] Ingested '{filename}': {added} chunks indexed")
        return added

    def ingest_file(self, path: Path) -> int:
        """Load a file from disk and ingest it."""
        return self.ingest_bytes(path.read_bytes(), path.name)

    def ingest_all(self) -> None:
        """Ingest every supported file in the documents directory."""
        for ext in ("*.pdf", "*.docx", "*.txt", "*.md"):
            for fpath in self._documents_path.glob(ext):
                try:
                    self.ingest_file(fpath)
                except Exception as exc:
                    print(f"[rag] Skipped '{fpath.name}': {exc}")

    # ------------------------------------------------------------------
    # Retrieval helpers
    # ------------------------------------------------------------------

    def _build_context(self, query: str) -> Tuple[str, List[str]]:
        """Return (context_text, source_list) for the given query."""
        if self._store.count() == 0:
            return "", []

        if _is_broad_query(query):
            return self._store.get_all()

        hits = self._store.hybrid_search(query)
        if not hits:
            return "", []

        parts:   List[str] = []
        sources: List[str] = []
        for doc, meta, _ in hits:
            src = meta.get("source", "document")
            sec = meta.get("section", "content")
            parts.append(f"[{src} — {sec}]\n{doc.strip()}")
            if src not in sources:
                sources.append(src)

        return "\n\n---\n\n".join(parts), sources

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def answer_question(
        self,
        question: str,
        top_k: int = 4,   # kept for API compat; actual count controlled by VectorStore
    ) -> Tuple[str, List[dict]]:
        """Non-streaming answer. Returns (answer_text, context_list)."""
        context_text, sources = self._build_context(question)

        contexts: List[dict] = [{"source": s} for s in sources]

        if not context_text:
            return (
                "I could not find relevant information in the uploaded documents.",
                contexts,
            )

        answer = self._llm.generate(context=context_text, question=question)
        return answer, contexts

    def answer_stream(
        self,
        question: str,
        history: List[dict] | None = None,
    ) -> Generator[str, None, None]:
        """Streaming answer. Yields text tokens as they arrive from Ollama.

        history — list of {"role": "user"|"assistant", "content": "..."} dicts
        """
        context_text, _ = self._build_context(question)
        yield from self._llm.generate_stream(
            context=context_text,
            question=question,
            history=history or [],
        )

    def clear_index(self) -> None:
        self._store.clear()

    def doc_count(self) -> int:
        return self._store.count()
