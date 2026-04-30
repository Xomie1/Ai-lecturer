"""
API & End-to-End tests — FastAPI endpoints
Author  : Madrine Nyawira Kariuki (Test Engineer)
Sprint  : 3 — Tasks T-3.05, T-3.06
Covers  : /health, /upload, /ask, /chat, /tts_status, /stt_status,
          /clear, /documents — response codes, payloads, edge cases
"""

import io
import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


# ── Health ────────────────────────────────────────────────────────────────────

class TestHealth:
    def test_health_returns_200(self):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_has_status_field(self):
        r = client.get("/health")
        assert "status" in r.json()

    def test_health_has_model_info(self):
        data = client.get("/health").json()
        assert "llm_model" in data
        assert "embed_model" in data


# ── Status endpoints ──────────────────────────────────────────────────────────

class TestStatusEndpoints:
    def test_tts_status_returns_bool(self):
        r = client.get("/tts_status")
        assert r.status_code == 200
        assert "available" in r.json()
        assert isinstance(r.json()["available"], bool)

    def test_stt_status_returns_bool(self):
        r = client.get("/stt_status")
        assert r.status_code == 200
        assert isinstance(r.json()["available"], bool)


# ── Upload ────────────────────────────────────────────────────────────────────

class TestUpload:
    def test_upload_unsupported_format_returns_400(self):
        r = client.post(
            "/upload",
            files={"file": ("test.exe", b"binary data", "application/octet-stream")},
        )
        assert r.status_code == 400

    def test_upload_empty_file_returns_400(self):
        r = client.post(
            "/upload",
            files={"file": ("empty.txt", b"", "text/plain")},
        )
        assert r.status_code == 400

    def test_upload_valid_txt_returns_200(self):
        content = (
            b"LECTURE NOTES\n\n"
            b"Artificial intelligence is the simulation of human intelligence.\n\n"
            b"Machine learning is a subset of artificial intelligence."
        )
        r = client.post(
            "/upload",
            files={"file": ("lecture.txt", content, "text/plain")},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ingested"
        assert data["chunks_added"] > 0


# ── Ask ───────────────────────────────────────────────────────────────────────

class TestAsk:
    def test_ask_returns_200(self):
        r = client.post("/ask", json={"question": "What is AI?"})
        assert r.status_code == 200

    def test_ask_returns_answer_text(self):
        r = client.post("/ask", json={"question": "What is machine learning?"})
        data = r.json()
        assert "answer_text" in data
        assert isinstance(data["answer_text"], str)
        assert len(data["answer_text"]) > 0

    def test_ask_returns_contexts_list(self):
        r = client.post("/ask", json={"question": "Tell me something."})
        assert "contexts" in r.json()
        assert isinstance(r.json()["contexts"], list)


# ── Chat session ──────────────────────────────────────────────────────────────

class TestChat:
    def test_chat_returns_event_stream(self):
        with client.stream(
            "POST", "/chat",
            data={"message": "Hello", "session_id": "test-session"},
        ) as r:
            assert r.status_code == 200
            assert "text/event-stream" in r.headers.get("content-type", "")

    def test_clear_session_returns_200(self):
        r = client.post("/clear", data={"session_id": "test-session"})
        assert r.status_code == 200
        assert r.json()["status"] == "cleared"


# ── Documents ─────────────────────────────────────────────────────────────────

class TestDocuments:
    def test_delete_documents_returns_200(self):
        r = client.delete("/documents")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_health_shows_zero_docs_after_delete(self):
        client.delete("/documents")
        r = client.get("/health")
        assert r.json()["docs_indexed"] == 0
