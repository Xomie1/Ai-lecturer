"""Vector store (ChromaDB) + BM25 index with Reciprocal Rank Fusion.

Architecture
------------
* ChromaDB  — persistent cosine-similarity vector search
* BM25Okapi — keyword-based sparse retrieval (rank_bm25)
* RRF       — fuses both ranked lists into a single ranked result

The combined retrieval is significantly better than either method alone,
especially on low-spec hardware where a smaller LLM benefits most from
high-quality context.
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import List, Optional, Tuple

import chromadb

from .embeddings import embed_text, embed_texts

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
COLLECTION_NAME = "documents"
BM25_PATH       = Path(__file__).parent.parent / "vector_db" / "bm25_index.pkl"
RRF_K           = 60   # RRF damping constant — higher = gentler rank weighting
HYBRID_N        = 12   # candidates to pull from each retriever before fusion
MAX_CONTEXT_CHARS = 40_000


# ---------------------------------------------------------------------------
# BM25 helpers
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    return re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()


class _BM25Index:
    """Thin wrapper around BM25Okapi that also persists to disk."""

    def __init__(self) -> None:
        self.corpus: List[str] = []
        self.ids:    List[str] = []
        self._index = None  # BM25Okapi | None

    # ------------------------------------------------------------------
    def _rebuild(self) -> None:
        from rank_bm25 import BM25Okapi
        if self.corpus:
            self._index = BM25Okapi([_tokenize(c) for c in self.corpus])
        else:
            self._index = None

    def add(self, chunks: List[str], ids: List[str]) -> None:
        self.corpus.extend(chunks)
        self.ids.extend(ids)
        self._rebuild()
        self._save()

    def search(self, query: str, n: int) -> List[str]:
        if self._index is None or not self.ids:
            return []
        scores  = self._index.get_scores(_tokenize(query))
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
        return [self.ids[i] for i in top_idx if scores[i] > 0]

    def clear(self) -> None:
        self.corpus, self.ids, self._index = [], [], None
        if BM25_PATH.exists():
            BM25_PATH.unlink()

    # ------------------------------------------------------------------
    def _save(self) -> None:
        BM25_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(BM25_PATH, "wb") as f:
            pickle.dump({"corpus": self.corpus, "ids": self.ids}, f)

    def load(self) -> None:
        if not BM25_PATH.exists():
            return
        try:
            with open(BM25_PATH, "rb") as f:
                data = pickle.load(f)
            self.corpus = data["corpus"]
            self.ids    = data["ids"]
            self._rebuild()
            print(f"[bm25] Loaded {len(self.corpus)} chunks from disk")
        except Exception as exc:
            print(f"[bm25] Load failed: {exc}")


# ---------------------------------------------------------------------------
# RRF fusion
# ---------------------------------------------------------------------------

def _rrf(vector_ids: List[str], bm25_ids: List[str]) -> List[str]:
    scores: dict[str, float] = {}
    for rank, doc_id in enumerate(vector_ids, 1):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (RRF_K + rank)
    for rank, doc_id in enumerate(bm25_ids, 1):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (RRF_K + rank)
    return sorted(scores, key=lambda x: scores[x], reverse=True)


# ---------------------------------------------------------------------------
# VectorStore — public interface
# ---------------------------------------------------------------------------

class VectorStore:
    """Manages ChromaDB collection and BM25 index together."""

    def __init__(self, persist_directory: Path) -> None:
        persist_directory.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(persist_directory))
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self._bm25 = _BM25Index()
        self._bm25.load()
        self._backfill_bm25()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add_chunks(
        self,
        chunks: List[str],
        filename: str,
    ) -> int:
        """Embed and store chunks. Returns number of chunks actually added."""
        added = 0
        new_chunks: List[str] = []
        new_ids:    List[str] = []

        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) < 30:
                continue
            chunk_id = f"{filename}-{i}-{hash(chunk) % 100_000}"
            emb      = embed_text(chunk)
            lines    = [l.strip() for l in chunk.split("\n") if l.strip()]
            section  = lines[0][:80] if lines and len(lines[0]) < 100 else "content"
            self._collection.add(
                ids=[chunk_id],
                embeddings=[emb],
                documents=[chunk],
                metadatas=[{
                    "source":    filename,
                    "section":   section,
                    "chunk_idx": i,
                    "char_count": len(chunk),
                }],
            )
            new_chunks.append(chunk)
            new_ids.append(chunk_id)
            added += 1

        if new_chunks:
            self._bm25.add(new_chunks, new_ids)

        return added

    def clear(self) -> None:
        """Delete all documents from ChromaDB and the BM25 index."""
        self._client.delete_collection(COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self._bm25.clear()

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def count(self) -> int:
        return self._collection.count()

    def hybrid_search(
        self,
        query: str,
        n: int = HYBRID_N,
    ) -> List[Tuple[str, dict, float]]:
        """Return (document_text, metadata, distance) tuples via BM25+vector+RRF."""
        doc_count = self.count()
        if doc_count == 0:
            return []

        # --- vector search ---
        qe = embed_text(query)
        # NOTE: "ids" is NOT a valid include value for .query() in ChromaDB —
        # IDs are always returned automatically in the result dict.
        v_results = self._collection.query(
            query_embeddings=[qe],
            n_results=min(n, doc_count),
            include=["documents", "metadatas", "distances"],
        )
        v_ids  = v_results["ids"][0]       if v_results.get("ids")       else []
        v_docs = v_results["documents"][0] if v_results.get("documents") else []
        v_meta = v_results["metadatas"][0] if v_results.get("metadatas") else []
        v_dist = v_results["distances"][0] if v_results.get("distances") else []

        # Build lookup from vector results
        lookup: dict[str, tuple[str, dict, float]] = {}
        for doc_id, doc, meta, dist in zip(v_ids, v_docs, v_meta, v_dist):
            lookup[doc_id] = (doc, meta or {}, dist)

        # --- BM25 search ---
        b_ids = self._bm25.search(query, n)

        # Fetch BM25 hits not already in vector results
        missing = [i for i in b_ids if i not in lookup]
        if missing:
            got = self._collection.get(
                ids=missing, include=["documents", "metadatas"]
            )
            for doc_id, doc, meta in zip(
                got.get("ids", []),
                got.get("documents", []),
                got.get("metadatas", []),
            ):
                lookup[doc_id] = (doc, meta or {}, 0.5)  # neutral distance

        # --- RRF fusion ---
        fused = _rrf(v_ids, b_ids)

        results: List[Tuple[str, dict, float]] = []
        source_counts: dict[str, int] = {}

        for doc_id in fused:
            if doc_id not in lookup:
                continue
            doc, meta, dist = lookup[doc_id]
            # Skip only very poor vector matches not rescued by BM25
            # 0.92 is intentionally lenient — slide-based PDFs produce high distances
            if dist > 0.92 and doc_id not in b_ids:
                continue
            # Cap chunks per source to avoid one document dominating
            src = meta.get("source", "document")
            if source_counts.get(src, 0) >= 5:
                continue
            source_counts[src] = source_counts.get(src, 0) + 1
            results.append((doc, meta, dist))

        return results

    def get_all(self) -> Tuple[str, List[str]]:
        """Return (combined_text, source_list) for broad / summary queries."""
        if self.count() == 0:
            return "", []
        result = self._collection.get(include=["documents", "metadatas"])
        parts, sources = [], []
        for doc, meta in zip(
            result.get("documents", []), result.get("metadatas", [])
        ):
            if doc and doc.strip():
                parts.append(doc.strip())
                src = (meta or {}).get("source", "document")
                if src not in sources:
                    sources.append(src)
        combined = "\n\n---\n\n".join(parts)
        return combined[:MAX_CONTEXT_CHARS], sources

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _backfill_bm25(self) -> None:
        """Populate BM25 from ChromaDB when BM25 index is missing but DB has docs."""
        if self._bm25.corpus:
            return
        count = self.count()
        if count == 0:
            return
        try:
            # IDs are always returned by .get() — "ids" is not a valid include value
            result = self._collection.get(include=["documents"])
            ids    = result.get("ids", [])
            docs   = result.get("documents", [])
            if ids and docs:
                self._bm25.add(docs, ids)
                print(f"[bm25] Backfilled {len(ids)} chunks from ChromaDB")
        except Exception as exc:
            print(f"[bm25] Backfill failed: {exc}")
