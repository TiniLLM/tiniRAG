"""Tests for tinirag.core.session."""

import json

import pytest

import tinirag.core.session as session_module
from tinirag.core.session import (
    append_history,
    list_sessions,
    load_session,
    new_session_id,
    save_session,
)


@pytest.fixture(autouse=True)
def patch_dirs(tmp_path, monkeypatch):
    sessions_dir = tmp_path / "sessions"
    history_file = tmp_path / "history.jsonl"
    monkeypatch.setattr(session_module, "SESSIONS_DIR", sessions_dir)
    monkeypatch.setattr(session_module, "HISTORY_FILE", history_file)


class TestNewSessionId:
    def test_format(self):
        # New format: YYYYMMDD-xxxxxxxx  (date prefix + hyphen + 8 hex chars)
        sid = new_session_id()
        parts = sid.split("-")
        assert len(parts) == 2
        assert len(parts[0]) == 8  # YYYYMMDD
        assert len(parts[1]) == 8  # uuid hex chars

    def test_date_prefix_is_digits(self):
        date_part = new_session_id().split("-")[0]
        assert date_part.isdigit()

    def test_unique(self):
        assert new_session_id() != new_session_id()


class TestSaveAndLoadSession:
    def test_save_creates_file(self, tmp_path):
        sid = "test1234"
        messages = [{"role": "user", "content": "Hello"}]
        path = save_session(sid, messages, title="Test session")
        assert path.exists()

    def test_load_returns_messages(self, tmp_path):
        sid = "abcd1234"
        messages = [
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "Python is a programming language."},
        ]
        save_session(sid, messages, title="Python chat")
        loaded = load_session(sid)
        assert loaded is not None
        assert loaded["messages"] == messages
        assert loaded["title"] == "Python chat"

    def test_load_missing_returns_none(self):
        assert load_session("nonexistent") is None

    def test_save_preserves_id(self):
        sid = "myid1234"
        save_session(sid, [], title="Empty")
        loaded = load_session(sid)
        assert loaded["id"] == sid

    def test_resave_preserves_created_timestamp(self):
        """Re-saving a resumed session must not overwrite the original creation time."""
        sid = "resave01"
        save_session(sid, [{"role": "user", "content": "first"}], title="First save")
        original_created = load_session(sid)["created"]

        # Simulate resume + re-save with additional messages
        save_session(
            sid,
            [{"role": "user", "content": "first"}, {"role": "user", "content": "second"}],
            title="Updated",
        )
        reloaded = load_session(sid)
        assert reloaded["created"] == original_created


class TestListSessions:
    def test_empty_returns_empty(self):
        assert list_sessions() == []

    def test_lists_saved_sessions(self):
        save_session("sess0001", [{"role": "user", "content": "Hi"}], title="First")
        save_session("sess0002", [{"role": "user", "content": "Hello"}], title="Second")
        sessions = list_sessions()
        ids = [s["id"] for s in sessions]
        assert "sess0001" in ids
        assert "sess0002" in ids

    def test_message_count_correct(self):
        msgs = [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
        ]
        save_session("counttest", msgs, title="Count")
        sessions = list_sessions()
        match = next(s for s in sessions if s["id"] == "counttest")
        assert match["message_count"] == 3


class TestAppendHistory:
    def test_appends_jsonl(self):
        import tinirag.core.session as sm

        append_history(
            raw_query="What is Python?",
            keywords="python",
            sources_used=["https://python.org"],
            response_length=42,
        )
        lines = sm.HISTORY_FILE.read_text().strip().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["raw_query"] == "What is Python?"
        assert entry["keywords"] == "python"
        assert entry["response_length"] == 42

    def test_multiple_appends(self):
        import tinirag.core.session as sm

        append_history("Q1", "kw1", [], 10)
        append_history("Q2", "kw2", [], 20)
        lines = sm.HISTORY_FILE.read_text().strip().splitlines()
        assert len(lines) == 2
