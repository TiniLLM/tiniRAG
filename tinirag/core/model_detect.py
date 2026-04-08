"""Auto-detect the best available Ollama/OpenAI-compatible model."""

from __future__ import annotations

import re

import httpx

# Process-level cache so we only query Ollama once per run
_CACHED_MODEL: str | None = None


def detect_available_model(endpoint_base: str) -> str | None:
    """Query {endpoint_base}/v1/models and return the best model for RAG.

    Returns None if Ollama is unreachable or no models are installed.
    Result is cached for the lifetime of the process.
    """
    global _CACHED_MODEL
    if _CACHED_MODEL is not None:
        return _CACHED_MODEL

    try:
        r = httpx.get(f"{endpoint_base}/v1/models", timeout=3.0)
        if r.status_code != 200:
            return None
        data = r.json()
        models = [m["id"] for m in data.get("data", [])]
    except Exception:
        return None

    if not models:
        return None

    result = _pick_best(models)
    _CACHED_MODEL = result
    return result


def _pick_best(models: list[str]) -> str:
    """Select the best model for RAG from a list of available model IDs.

    Priority:
    1. Prefer instruct/chat-tuned models (name contains instruct, chat, :it, -it).
    2. Among those, prefer the smallest that is >= 3b parameters.
    3. If no parameter count is parseable, take the first instruct/chat model.
    4. If no instruct/chat models exist, apply same size logic to all models.
    5. Final fallback: first model in the list.
    """
    _INSTRUCT_MARKERS = ("instruct", "chat", ":it", "-it")

    def is_instruct(name: str) -> bool:
        lower = name.lower()
        return any(marker in lower for marker in _INSTRUCT_MARKERS)

    def param_billions(name: str) -> float | None:
        """Parse parameter size from name like llama3.2:3b → 3.0, qwen2.5:14b → 14.0."""
        m = re.search(r"(\d+(?:\.\d+)?)b", name.lower())
        return float(m.group(1)) if m else None

    instruct_models = [m for m in models if is_instruct(m)]
    candidates = instruct_models if instruct_models else models

    # Find smallest model >= 3b
    sized = [(m, param_billions(m)) for m in candidates]
    eligible = [(m, sz) for m, sz in sized if sz is not None and sz >= 3.0]

    if eligible:
        return min(eligible, key=lambda x: x[1])[0]

    # No parseable sizes — return first candidate
    return candidates[0]


def _reset_cache() -> None:
    """Reset the module-level cache. Used in tests."""
    global _CACHED_MODEL
    _CACHED_MODEL = None
