# AI Lecture Explainer 
---

## What This Project Does

A fully **local, on-premise** AI document query system built for NexusGuard.

- Upload a PDF, DOCX, TXT, or MD lecture file
- Ask questions about it in natural language
- Receive a **streaming text answer** grounded in the document
- Hear the answer **spoken aloud** via Kokoro TTS
- Use your **microphone** to ask questions hands-free via Whisper STT

**Zero external API calls.** Everything runs on your machine via Ollama.

---

## Quick Start

### Prerequisites
- [Ollama](https://ollama.com/download) installed and on PATH
- Python 3.10+ (Miniconda recommended)
- ~3 GB free disk space (for models)

### Setup

```bash
# 1. Create and activate environment
conda create -n ai-lecture python=3.11 -y
conda activate ai-lecture

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app (Ollama starts automatically)
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Then open **http://localhost:8000** in your browser.

On first run, the app will automatically pull:
- `qwen2.5:3b` (~2 GB) — language model
- `nomic-embed-text` (~270 MB) — embedding model

---

## Usage

1. **Upload** — Drag a PDF/DOCX/TXT/MD file onto the left sidebar
2. **Ask** — Type a question and press Enter (or Shift+Enter for newline)
3. **Listen** — Toggle 🔊 to auto-play answers as speech
4. **Record** — Hold 🎙 to record a voice question; release to transcribe

---

## Project Structure

```
ai-lecture-explainer/
│
├── backend/
│   ├── main.py              # FastAPI app — all HTTP endpoints
│   ├── rag_pipeline.py      # Retrieval orchestration
│   ├── document_loader.py   # PDF/DOCX/TXT extraction + chunking
│   ├── embeddings.py        # nomic-embed-text via Ollama
│   ├── vector_store.py      # ChromaDB + BM25 + RRF hybrid search
│   └── llama_model.py       # qwen2.5:3b via Ollama (streaming)
│
├── frontend/
│   ├── index.html           # Single-page chat UI
│   └── styles.css           # Responsive stylesheet + branding
│
├── tests/
│   ├── test_document_loader.py   # Unit tests — extraction & chunking
│   ├── test_rag_pipeline.py      # Integration tests — retrieval
│   └── test_api.py               # API & end-to-end tests
│
├── docs/
│   └── architecture.md      # System design & component overview
│
├── vector_db/               # ChromaDB persistent storage (auto-created)
├── data/documents/          # Uploaded files (auto-created)
├── requirements.txt
└── README.md
```

---

## Running Tests

```bash
# From ai-lecture-explainer/
pytest tests/ -v
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Web UI |
| GET | `/health` | Ollama status + index info |
| POST | `/upload` | Ingest a document |
| POST | `/ask` | Single-shot Q&A (JSON) |
| POST | `/chat` | Streaming Q&A (SSE) |
| POST | `/speak` | Text → WAV audio |
| POST | `/transcribe` | Audio → transcript text |
| GET | `/tts_status` | Kokoro availability |
| GET | `/stt_status` | Whisper availability |
| POST | `/clear` | Clear session history |
| DELETE | `/documents` | Wipe entire index |

