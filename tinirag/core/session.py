"""Chat session persistence: save/load/list interactive REPL sessions."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from tinirag.config import HISTORY_FILE, SESSIONS_DIR


def _ensure_dirs() -> None:
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)


def new_session_id() -> str:
    # 12 hex chars = 48 bits; birthday collision probability <1% up to ~20 M sessions.
    # Prefix with compact UTC date so IDs sort chronologically and are human-readable.
    date_prefix = datetime.now(timezone.utc).strftime("%Y%m%d")
    return f"{date_prefix}-{uuid.uuid4().hex[:8]}"


def save_session(session_id: str, messages: list[dict], title: str = "") -> Path:
    """Save a chat session as JSON under ~/.tinirag/sessions/."""
    _ensure_dirs()
    path = SESSIONS_DIR / f"{session_id}.json"

    # Preserve the original creation timestamp when re-saving a resumed session
    if path.exists():
        try:
            existing = json.loads(path.read_text())
            created = existing.get("created") or datetime.now(timezone.utc).isoformat()
        except Exception:
            created = datetime.now(timezone.utc).isoformat()
    else:
        created = datetime.now(timezone.utc).isoformat()

    data = {
        "id": session_id,
        "title": title or f"Session {session_id}",
        "created": created,
        "messages": messages,
    }
    path.write_text(json.dumps(data, indent=2))
    return path


def load_session(session_id: str) -> dict | None:
    _ensure_dirs()
    path = SESSIONS_DIR / f"{session_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def list_sessions() -> list[dict]:
    """Return all saved sessions sorted by creation time (newest first)."""
    _ensure_dirs()
    sessions = []
    for f in sorted(SESSIONS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            data = json.loads(f.read_text())
            sessions.append(
                {
                    "id": data.get("id", f.stem),
                    "title": data.get("title", f.stem),
                    "created": data.get("created", ""),
                    "message_count": len(data.get("messages", [])),
                }
            )
        except Exception:
            continue
    return sessions


def append_history(
    raw_query: str,
    keywords: str,
    sources_used: list[str],
    response_length: int,
) -> None:
    """Opt-in query history: append one JSONL line to ~/.tinirag/history.jsonl."""
    _ensure_dirs()
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "raw_query": raw_query,
        "keywords": keywords,
        "sources_used": sources_used,
        "response_length": response_length,
    }
    with open(HISTORY_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
