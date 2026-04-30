"""AI Lecture Explainer — FastAPI backend.

Endpoints
---------
GET  /              → health / info
GET  /health        → Ollama + index status
POST /upload        → ingest PDF / DOCX / TXT / MD
POST /ask           → single-shot Q&A (blocking, returns JSON)
POST /chat          → streaming Q&A via SSE (text/event-stream)
POST /clear         → clear chat history for a session
DELETE /documents   → wipe the entire vector index

POST /speak         → Kokoro TTS  → returns WAV audio
POST /transcribe    → Whisper STT → returns transcript text
GET  /tts_status    → {"available": bool}
GET  /stt_status    → {"available": bool}
"""

from __future__ import annotations

import asyncio
import functools
import io
import json
import os
import re
import tempfile
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Paths & pipeline
# ---------------------------------------------------------------------------
_BASE_DIR    = Path(__file__).resolve().parents[1]
_DATA_DIR    = _BASE_DIR / "data" / "documents"
_VECTOR_DIR  = _BASE_DIR / "vector_db"
_TEMPLATE    = _BASE_DIR / "frontend" / "index.html"

_DATA_DIR.mkdir(parents=True, exist_ok=True)
_VECTOR_DIR.mkdir(parents=True, exist_ok=True)

from .rag_pipeline import RagPipeline

# ---------------------------------------------------------------------------
# Auto-start Ollama if it isn't already running
# ---------------------------------------------------------------------------
import shutil
import subprocess
import time

def _ensure_ollama_running() -> None:
    """Start `ollama serve` in the background if the daemon isn't up yet."""
    import ollama as _ol
    # Quick ping — if it responds, we're done
    try:
        _ol.list()
        print("[ollama] Already running ✅")
        return
    except Exception:
        pass

    exe = shutil.which("ollama")
    if not exe:
        print("[ollama] ⚠️  'ollama' not found on PATH. Install from https://ollama.com")
        return

    print("[ollama] Starting ollama serve…")
    subprocess.Popen(
        [exe, "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        # On Windows, detach so it keeps running if this terminal closes
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
    )

    # Wait up to 15 s for the daemon to become ready
    for attempt in range(15):
        time.sleep(1)
        try:
            _ol.list()
            print(f"[ollama] Ready after {attempt + 1}s ✅")
            return
        except Exception:
            pass

    print("[ollama] ⚠️  Ollama didn't respond after 15 s — continuing anyway.")

_ensure_ollama_running()

# ---------------------------------------------------------------------------
# Auto-pull required Ollama models if not already present
# ---------------------------------------------------------------------------
_REQUIRED_MODELS = ["qwen2.5:3b", "nomic-embed-text"]

def _ensure_ollama_models() -> None:
    try:
        import ollama
        available = {m["model"] for m in (ollama.list().get("models") or [])}
        for model in _REQUIRED_MODELS:
            already_have = any(m.startswith(model) for m in available)
            if already_have:
                print(f"[ollama] {model} already present ✅")
            else:
                print(f"[ollama] Pulling {model} — this may take a few minutes on first run…")
                ollama.pull(model)
                print(f"[ollama] {model} ready ✅")
    except Exception as exc:
        print(f"[ollama] Could not auto-pull models: {exc}")

_ensure_ollama_models()

_pipeline = RagPipeline(
    documents_path=_DATA_DIR,
    vector_db_path=_VECTOR_DIR,
)

# Pre-warm the LLM so the first demo query isn't cold-start slow
def _warm_llm() -> None:
    try:
        import ollama
        ollama.chat(
            model="qwen2.5:3b",
            messages=[{"role": "user", "content": "hi"}],
            stream=False,
            keep_alive="30m",
            options={"num_predict": 1, "num_ctx": 512},
        )
        print("[llm] Model pre-warmed ✅")
    except Exception as exc:
        print(f"[llm] Pre-warm skipped: {exc}")

import threading
threading.Thread(target=_warm_llm, daemon=True).start()

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="AI Lecture Explainer", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static assets (styles.css, any future JS/image files)
_STATIC_DIR = _BASE_DIR / "frontend"
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

# ---------------------------------------------------------------------------
# Whisper STT  — lazy-loaded on first /transcribe call
# ---------------------------------------------------------------------------
_whisper_model = None
_whisper_lock  = asyncio.Lock()
_whisper_ok: bool = False

def _try_import_whisper() -> bool:
    try:
        import whisper  # noqa: F401
        return True
    except ImportError:
        return False

_whisper_ok = _try_import_whisper()
if _whisper_ok:
    print("✅ Whisper STT found — /transcribe endpoint active.")
else:
    print("⚠️  Whisper not installed.  pip install openai-whisper")


async def _get_whisper():
    global _whisper_model
    if not _whisper_ok:
        return None
    if _whisper_model is not None:
        return _whisper_model
    async with _whisper_lock:
        if _whisper_model is None:
            try:
                import whisper as _w
                print("  [STT] Loading Whisper 'base'…")
                loop = asyncio.get_running_loop()
                _whisper_model = await loop.run_in_executor(
                    None, functools.partial(_w.load_model, "base")
                )
                print("  [STT] Whisper ready ✅")
            except Exception as exc:
                print(f"  [STT] Failed: {exc}")
    return _whisper_model


# ---------------------------------------------------------------------------
# Kokoro TTS  — lazy-loaded on first /speak call
# ---------------------------------------------------------------------------
_kokoro_pipeline = None
_kokoro_lock     = asyncio.Lock()
_kokoro_ok: bool = False
TTS_VOICE        = "af_heart"    # warm American-English female voice

def _try_import_kokoro() -> bool:
    try:
        import kokoro  # noqa: F401
        return True
    except ImportError:
        return False

_kokoro_ok = _try_import_kokoro()
if _kokoro_ok:
    print("✅ Kokoro TTS found — /speak endpoint active.")
else:
    print("⚠️  Kokoro not installed.  pip install kokoro soundfile")


async def _get_kokoro():
    global _kokoro_pipeline
    if not _kokoro_ok:
        return None
    if _kokoro_pipeline is not None:
        return _kokoro_pipeline
    async with _kokoro_lock:
        if _kokoro_pipeline is None:
            try:
                from kokoro import KPipeline
                print(f"  [TTS] Loading Kokoro (voice={TTS_VOICE})…")
                _kokoro_pipeline = KPipeline(lang_code="a")   # 'a' = American English
                print("  [TTS] Kokoro ready ✅")
            except Exception as exc:
                print(f"  [TTS] Failed: {exc}")
    return _kokoro_pipeline


def _strip_markdown(text: str) -> str:
    """Strip markdown symbols so TTS doesn't read them aloud."""
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*",     r"\1", text)
    text = re.sub(r"`([^`]*)`",     r"\1", text)
    text = re.sub(r"#{1,6}\s*",     "",    text)
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    text = re.sub(r"[•›\-]\s+",    ", ",  text)
    text = re.sub(r"\s+",           " ",   text)
    return text.strip()


# ---------------------------------------------------------------------------
# In-memory chat history  {session_id: [{"role": ..., "content": ...}, ...]}
# ---------------------------------------------------------------------------
_chat_history: dict[str, list[dict]] = {}


# ---------------------------------------------------------------------------
# Routes — info / health
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def get_ui():
    if _TEMPLATE.exists():
        return HTMLResponse(_TEMPLATE.read_text(encoding="utf-8"))
    return HTMLResponse(
        "<h2>AI Lecture Explainer API</h2>"
        "<p>See <a href='/docs'>/docs</a> for the interactive API.</p>"
    )


@app.get("/health")
async def health():
    ollama_ok = False
    try:
        import ollama
        ollama.list()
        ollama_ok = True
    except Exception:
        pass

    return {
        "status":       "ok" if ollama_ok else "degraded",
        "ollama":       "connected" if ollama_ok else "unreachable",
        "llm_model":    "qwen2.5:3b",
        "embed_model":  "nomic-embed-text",
        "tts":          "kokoro" if _kokoro_ok else "unavailable",
        "stt":          "whisper" if _whisper_ok else "unavailable",
        "docs_indexed": _pipeline.doc_count(),
        "sessions":     len(_chat_history),
    }


# ---------------------------------------------------------------------------
# Routes — document ingestion
# ---------------------------------------------------------------------------

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    name = (file.filename or "").lower()
    if not any(name.endswith(ext) for ext in (".pdf", ".docx", ".txt", ".md")):
        raise HTTPException(400, "Supported formats: PDF, DOCX, TXT, MD")

    content = await file.read()
    if not content:
        raise HTTPException(400, "Empty file")

    try:
        added = _pipeline.ingest_bytes(content, file.filename)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        print(f"[upload] {exc}")
        raise HTTPException(500, str(exc))

    return {
        "status":       "ingested",
        "filename":     file.filename,
        "chunks_added": added,
    }


@app.delete("/documents")
async def delete_all_documents():
    try:
        _pipeline.clear_index()
        _chat_history.clear()
        return {"status": "ok", "message": "All documents and sessions cleared."}
    except Exception as exc:
        raise HTTPException(500, str(exc))


# ---------------------------------------------------------------------------
# Routes — Q&A
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    question: str
    top_k:    int = 4


@app.post("/ask")
async def ask_question(payload: AskRequest):
    """Blocking single-shot Q&A. Good for simple integrations."""
    answer, contexts = _pipeline.answer_question(payload.question, payload.top_k)
    return {"answer_text": answer, "contexts": contexts}


@app.post("/chat")
async def chat(
    message:    str = Form(...),
    session_id: str = Form(default="default"),
):
    """Streaming Q&A via Server-Sent Events.

    Each SSE event is JSON with one of:
      {"type": "token",  "content": "<text>"}
      {"type": "done",   "sources": [...], "docs_indexed": N}
    """
    if session_id not in _chat_history:
        _chat_history[session_id] = []

    history = _chat_history[session_id]

    def _stream():
        full_response = ""
        sources_used: list[str] = []

        # --- retrieve context separately so we can report sources ---
        from .rag_pipeline import _is_broad_query
        from .vector_store import MAX_CONTEXT_CHARS

        context_text = ""
        doc_count    = _pipeline.doc_count()

        if doc_count > 0:
            try:
                if _is_broad_query(message):
                    context_text, sources_used = _pipeline._store.get_all()
                else:
                    hits = _pipeline._store.hybrid_search(message)
                    parts: list[str] = []
                    for doc, meta, _ in hits:
                        src = meta.get("source", "document")
                        sec = meta.get("section", "content")
                        parts.append(f"[{src} — {sec}]\n{doc.strip()}")
                        if src not in sources_used:
                            sources_used.append(src)
                    context_text = "\n\n---\n\n".join(parts)
            except Exception as exc:
                print(f"[rag] {exc}")

        # --- stream tokens from LLM ---
        try:
            for token in _pipeline._llm.generate_stream(
                context=context_text,
                question=message,
                history=history[-10:],
            ):
                full_response += token
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
        except Exception as exc:
            err = f"Streaming error: {exc}"
            full_response = err
            yield f"data: {json.dumps({'type': 'token', 'content': err})}\n\n"

        # --- update history ---
        history.append({"role": "user",      "content": message})
        history.append({"role": "assistant", "content": full_response})
        if len(history) > 10:
            _chat_history[session_id] = history[-10:]

        yield f"data: {json.dumps({'type': 'done', 'sources': sources_used, 'docs_indexed': doc_count})}\n\n"

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/clear")
async def clear_chat(session_id: str = Form(default="default")):
    _chat_history.pop(session_id, None)
    return {"status": "cleared"}


# ---------------------------------------------------------------------------
# Routes — TTS (Kokoro)
# ---------------------------------------------------------------------------

@app.get("/tts_status")
async def tts_status():
    return {"available": _kokoro_ok}


@app.post("/speak")
async def speak(text: str = Form(...)):
    """Convert text to speech using Kokoro and return a WAV file."""
    if not _kokoro_ok:
        raise HTTPException(
            503, "Kokoro not installed. Run: pip install kokoro soundfile"
        )

    pipeline = await _get_kokoro()
    if pipeline is None:
        raise HTTPException(503, "Kokoro failed to load — check server logs.")

    clean = _strip_markdown(text)
    if not clean:
        raise HTTPException(400, "No speakable text provided.")

    # Limit TTS to first ~300 chars (≈2 sentences) so playback starts fast on CPU
    sentences = re.split(r'(?<=[.!?])\s+', clean)
    preview = ""
    for s in sentences:
        if len(preview) + len(s) > 300:
            break
        preview += (" " if preview else "") + s
    clean = preview or clean[:300]

    try:
        import soundfile as sf

        audio_chunks = []
        for result in pipeline(clean, voice=TTS_VOICE, speed=1.0, split_pattern=r"\n+"):
            audio = result.audio
            if audio is not None and len(audio) > 0:
                audio_chunks.append(audio)

        if not audio_chunks:
            raise HTTPException(500, "Kokoro produced no audio.")

        combined    = np.concatenate(audio_chunks)
        sample_rate = 24_000   # Kokoro outputs 24 kHz

        buf = io.BytesIO()
        sf.write(buf, combined, sample_rate, format="WAV")
        wav_bytes = buf.getvalue()

        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={
                "Content-Length": str(len(wav_bytes)),
                "Cache-Control":  "no-cache",
            },
        )
    except HTTPException:
        raise
    except Exception as exc:
        print(f"[tts] {exc}")
        raise HTTPException(500, f"TTS failed: {exc}")


# ---------------------------------------------------------------------------
# Routes — STT (Whisper)
# ---------------------------------------------------------------------------

@app.get("/stt_status")
async def stt_status():
    return {"available": _whisper_ok}


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    """Transcribe an audio recording using Whisper and return the text."""
    if not _whisper_ok:
        raise HTTPException(
            503, "Whisper not installed. Run: pip install openai-whisper"
        )

    content = await audio.read()
    if not content:
        raise HTTPException(400, "Empty audio file.")

    ct = (audio.content_type or "").lower()
    if "mp4" in ct or "m4a" in ct:
        suffix = ".mp4"
    elif "ogg" in ct:
        suffix = ".ogg"
    elif "wav" in ct:
        suffix = ".wav"
    else:
        suffix = ".webm"   # Chrome / Firefox default

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        model = await _get_whisper()
        if model is None:
            raise HTTPException(503, "Whisper failed to load — check server logs.")

        loop   = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, functools.partial(model.transcribe, tmp_path, fp16=False)
        )
        text = (result.get("text") or "").strip()
        if not text:
            raise HTTPException(422, "No speech detected in the recording.")
        return {"text": text}

    except HTTPException:
        raise
    except Exception as exc:
        print(f"[stt] {exc}")
        raise HTTPException(500, f"Transcription failed: {exc}")
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
