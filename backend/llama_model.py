"""LLM integration via Ollama — qwen2.5:3b.

qwen2.5:3b is chosen deliberately for the target hardware (HP 14-ep0xxx,
low-grade Mac — both CPU-only).  It fits comfortably in ~2 GB of RAM and
runs at an acceptable speed (8–15 tok/s on CPU), while still delivering
solid instruction-following for Q&A over lecture documents.

To swap model, change DEFAULT_MODEL or pass model_name= to LlamaModel().
"""

from __future__ import annotations

from typing import Generator, List

DEFAULT_MODEL = "qwen2.5:3b"

_SYSTEM = "You are a document question-answering assistant. Answer only from the document context given in each message."

_GENERAL_SYSTEM = "You are a helpful AI assistant. Answer questions clearly and concisely."

_OLLAMA_OPTIONS = {
    "temperature": 0.1,   # lower = more faithful to context
    "num_predict": 400,   # keeps answers concise; faster on CPU
    "top_p":       0.9,
    "num_ctx":     4096,  # halved from 8192 — sufficient for 3B, saves RAM
}


def _build_messages(
    context: str,
    question: str,
    history: List[dict],
) -> List[dict]:
    if context:
        # Inject context directly into the user turn — small models (3B)
        # are far more likely to obey instructions placed here than in
        # the system prompt alone.
        user_content = (
            f"Use ONLY the document excerpts below to answer. "
            f"Do NOT use outside knowledge. "
            f"If the answer is not in the excerpts, say: "
            f"\"That information is not in the uploaded document.\"\n\n"
            f"--- DOCUMENT EXCERPTS ---\n{context}\n--- END ---\n\n"
            f"Question: {question}"
        )
        messages = [{"role": "system", "content": _SYSTEM}]
    else:
        user_content = question
        messages = [{"role": "system", "content": _GENERAL_SYSTEM}]

    messages.extend(history[-10:])
    messages.append({"role": "user", "content": user_content})
    return messages


class LlamaModel:
    """Thin wrapper around ollama.chat / ollama.generate."""

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self._model = model_name

    # ------------------------------------------------------------------
    # Non-streaming (used by /ask endpoint)
    # ------------------------------------------------------------------

    def generate(self, context: str, question: str) -> str:
        """Return a complete answer string."""
        try:
            import ollama
            messages = _build_messages(context, question, [])
            response = ollama.chat(
                model=self._model,
                messages=messages,
                stream=False,
                keep_alive="30m",
                options=_OLLAMA_OPTIONS,
            )
            return (response.get("message") or {}).get("content", "").strip()
        except Exception as exc:
            return (
                f"Could not generate an answer. Make sure Ollama is running "
                f"('ollama serve') and the model is pulled "
                f"('ollama pull {self._model}'). Error: {exc}"
            )

    # ------------------------------------------------------------------
    # Streaming (used by /chat endpoint)
    # ------------------------------------------------------------------

    def generate_stream(
        self,
        context: str,
        question: str,
        history: List[dict] | None = None,
    ) -> Generator[str, None, None]:
        """Yield text tokens as they arrive from Ollama."""
        try:
            import ollama
            messages = _build_messages(context, question, history or [])
            for chunk in ollama.chat(
                model=self._model,
                messages=messages,
                stream=True,
                keep_alive="30m",
                options=_OLLAMA_OPTIONS,
            ):
                token = (chunk.get("message") or {}).get("content") or ""
                if token:
                    yield token
        except Exception as exc:
            yield (
                f"Error reaching Ollama. Is it running? "
                f"Try `ollama serve`. Details: {exc}"
            )
