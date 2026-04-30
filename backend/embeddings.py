"""Embedding generation via Ollama (nomic-embed-text).

Replaces the old HuggingFace sentence-transformers dependency so that
everything runs through a single Ollama process — no separate model
download or runtime required.
"""

from __future__ import annotations

from typing import List

EMBED_MODEL = "nomic-embed-text"


def embed_text(text: str) -> List[float]:
    """Embed a single string and return the vector."""
    import ollama
    return ollama.embeddings(model=EMBED_MODEL, prompt=text)["embedding"]


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a list of strings. Returns a list of vectors in the same order."""
    return [embed_text(t) for t in texts]
