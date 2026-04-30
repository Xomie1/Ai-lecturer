# System Architecture — AI Lecture Explainer

**Author:** Yugendhar Reddy Kommula (Software Developer)
**Sprint:** 1 — Task T-1.03 | Sprint 3 — Task T-3.08

---

## Overview

NexusGuard's AI Lecture Explainer is a fully **on-premise, zero-API** document query system.
Every component — embedding, retrieval, language generation, and speech synthesis — runs
locally via Ollama. No data leaves the machine.

---

## Architecture Diagram

```
Browser (index.html + styles.css)
        │
        │  HTTP / SSE
        ▼
FastAPI Backend  (backend/main.py)
        │
        ├── POST /upload ──► document_loader.py
        │                        │  extract + chunk
        │                        ▼
        │                    embeddings.py  ──► Ollama nomic-embed-text
        │                        │
        │                        ▼
        │                    vector_store.py  ──► ChromaDB  (disk)
        │                                    ──► BM25 index (disk)
        │
        └── POST /chat  ──► rag_pipeline.py
                                │
                                ├── hybrid_search()
                                │     ├── vector search (ChromaDB)
                                │     ├── keyword search (BM25)
                                │     └── RRF fusion
                                │
                                └── llama_model.py ──► Ollama qwen2.5:3b
                                                            │
                                                            ▼
                                                    SSE token stream
                                                            │
                                                            ▼
                                                    Kokoro TTS (/speak)
                                                            │
                                                            ▼
                                                    WAV audio → browser
```

---

## Component Breakdown

| File | Responsibility | Sprint Task |
|---|---|---|
| `backend/main.py` | FastAPI routes, TTS, STT, auto-start Ollama | T-3.01, T-3.02 |
| `backend/rag_pipeline.py` | Orchestrates retrieval + LLM generation | T-2.01, T-2.02 |
| `backend/document_loader.py` | PDF/DOCX/TXT/MD extraction + smart chunking | T-1.04 |
| `backend/embeddings.py` | nomic-embed-text via Ollama | T-1.05 |
| `backend/vector_store.py` | ChromaDB + BM25 + RRF hybrid search | T-1.06 |
| `backend/llama_model.py` | qwen2.5:3b via Ollama, streaming + non-streaming | T-2.01 |
| `frontend/index.html` | Chat UI, file upload, mic, audio playback | T-2.06, T-3.03 |
| `frontend/styles.css` | Responsive layout, Trivera Labs branding | T-2.07 |
| `tests/` | Unit, integration, API, and end-to-end tests | T-1.07, T-2.04, T-3.05 |

---

## Retrieval Strategy

Queries go through **two parallel searches** before reaching the LLM:

1. **Vector search** — query is embedded with `nomic-embed-text`; ChromaDB returns top-N semantically similar chunks
2. **BM25 keyword search** — same query is tokenised; BM25Okapi returns top-N keyword-matching chunks
3. **RRF Fusion** — Reciprocal Rank Fusion merges both ranked lists; chunks appearing in both searches rank highest

This hybrid approach outperforms single-method retrieval, especially on:
- Keyword-specific questions: *"What is Task T-2.03?"*
- Semantic/paraphrased questions: *"Who handles audio output?"*

---

## Models Used

| Model | Purpose | Size | Runtime |
|---|---|---|---|
| `qwen2.5:3b` | Language generation | ~2 GB | Ollama (CPU) |
| `nomic-embed-text` | Text embeddings | ~270 MB | Ollama (CPU) |
| `Kokoro (af_heart)` | Text-to-speech | ~82 MB | Local (CPU) |
| `Whisper base` | Speech-to-text | ~150 MB | Local (CPU) |

All models are CPU-compatible — no GPU required.

---

## Data Flow — Upload

```
User selects file
      │
      ▼
POST /upload  ─► extract_text()  ─► clean_text()  ─► split_into_chunks()
                                                            │
                                          ┌─────────────────┘
                                          │
                                          ▼
                               embed_text() via Ollama nomic-embed-text
                                          │
                               ┌──────────┴──────────┐
                               ▼                     ▼
                          ChromaDB.add()        BM25Index.add()
                          (vectors on disk)     (pickle on disk)
```

## Data Flow — Query

```
User types question
      │
      ▼
POST /chat (SSE)
      │
      ├── hybrid_search(question)
      │         ├── embed_text(question) → ChromaDB.query()
      │         ├── BM25.search(question)
      │         └── rrf(vector_ids, bm25_ids) → top chunks
      │
      └── llama_model.generate_stream(context=chunks, question)
                │  [SSE tokens stream to browser]
                ▼
          Kokoro TTS → WAV → browser auto-plays
```

---

## Known Limitations

- **CPU-only performance**: On machines without a GPU, `qwen2.5:3b` runs at ~8–15 tokens/sec. Larger models are not recommended.
- **Image-only PDFs**: PDFs that are purely scanned images (no embedded text) will produce no chunks. OCR support is not included.
- **In-memory session history**: Chat history is stored in RAM and lost on server restart.
- **Whisper first-call latency**: The `base` Whisper model takes ~10 seconds to load on first transcription.
