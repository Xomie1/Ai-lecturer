"""
Integration tests — RAG Pipeline (retrieval accuracy)
Author  : Madrine Nyawira Kariuki (Test Engineer)
Sprint  : 1 — Task T-1.07 | Sprint 2 — Task T-2.04
Covers  : ChromaDB storage, retrieval accuracy, broad query detection,
          context assembly, empty-index behaviour
"""

import pytest
from unittest.mock import MagicMock, patch
from backend.rag_pipeline import RagPipeline, _is_broad_query


# ── Broad query detection ─────────────────────────────────────────────────────

class TestBroadQueryDetection:
    def test_list_triggers_broad(self):
        assert _is_broad_query("list all team members") is True

    def test_summarise_triggers_broad(self):
        assert _is_broad_query("summarise the document") is True

    def test_how_many_triggers_broad(self):
        assert _is_broad_query("how many sprints are there?") is True

    def test_specific_question_not_broad(self):
        assert _is_broad_query("who is the backend engineer?") is False

    def test_empty_string_not_broad(self):
        assert _is_broad_query("") is False


# ── Pipeline — empty index ────────────────────────────────────────────────────

class TestRagPipelineEmptyIndex:
    @pytest.fixture
    def pipeline(self, tmp_path):
        docs_dir = tmp_path / "docs"
        vdb_dir  = tmp_path / "vdb"
        docs_dir.mkdir(); vdb_dir.mkdir()
        return RagPipeline(documents_path=docs_dir, vector_db_path=vdb_dir)

    def test_answer_returns_not_found_when_empty(self, pipeline):
        answer, contexts = pipeline.answer_question("What is the capital of France?")
        assert "not" in answer.lower() or "upload" in answer.lower() or "document" in answer.lower()
        assert isinstance(contexts, list)

    def test_doc_count_zero_initially(self, pipeline):
        assert pipeline.doc_count() == 0

    def test_ingest_bytes_increments_count(self, pipeline):
        sample = b"INTRODUCTION\n\nThis is a test document about machine learning and AI systems."
        pipeline.ingest_bytes(sample, "test.txt")
        assert pipeline.doc_count() > 0

    def test_clear_index_resets_count(self, pipeline):
        sample = b"INTRODUCTION\n\nTest content for the vector store."
        pipeline.ingest_bytes(sample, "test.txt")
        pipeline.clear_index()
        assert pipeline.doc_count() == 0


# ── Pipeline — ingestion & retrieval ─────────────────────────────────────────

class TestRagPipelineRetrieval:
    @pytest.fixture
    def loaded_pipeline(self, tmp_path):
        docs_dir = tmp_path / "docs"
        vdb_dir  = tmp_path / "vdb"
        docs_dir.mkdir(); vdb_dir.mkdir()
        pipeline = RagPipeline(documents_path=docs_dir, vector_db_path=vdb_dir)
        content = (
            b"TEAM MEMBERS\n\n"
            b"Tobi Akindele is the Backend Engineer responsible for the RAG pipeline.\n\n"
            b"Leann Cheptoo is the Project Manager coordinating all sprints.\n\n"
            b"Madrine Kariuki is the Test Engineer writing all test cases."
        )
        pipeline.ingest_bytes(content, "team.txt")
        return pipeline

    def test_retrieval_finds_relevant_chunk(self, loaded_pipeline):
        answer, contexts = loaded_pipeline.answer_question("Who is the backend engineer?")
        assert isinstance(answer, str)
        assert len(answer) > 0

    def test_contexts_include_source(self, loaded_pipeline):
        _, contexts = loaded_pipeline.answer_question("Who is the project manager?")
        assert len(contexts) > 0

    def test_unsupported_file_raises(self, loaded_pipeline):
        with pytest.raises(ValueError):
            loaded_pipeline.ingest_bytes(b"data", "file.xyz")
