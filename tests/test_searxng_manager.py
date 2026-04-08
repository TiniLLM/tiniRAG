"""Tests for tinirag.core.searxng_manager (all subprocess and HTTP mocked)."""

from __future__ import annotations

import signal
from unittest.mock import MagicMock, patch

import pytest

import tinirag.core.searxng_manager as mgr


@pytest.fixture(autouse=True)
def patch_paths(tmp_path, monkeypatch):
    """Redirect all file paths to tmp_path so tests never touch ~/.tinirag."""
    searxng_dir = tmp_path / "searxng"
    monkeypatch.setattr(mgr, "SEARXNG_DIR", searxng_dir)
    monkeypatch.setattr(mgr, "SETTINGS_FILE", searxng_dir / "settings.yml")
    monkeypatch.setattr(mgr, "SEARXNG_PID_FILE", tmp_path / "searxng.pid")
    monkeypatch.setattr(mgr, "SEARXNG_LOG_FILE", tmp_path / "searxng.log")


# ---------------------------------------------------------------------------
# Settings management
# ---------------------------------------------------------------------------


class TestEnsureSettings:
    def test_copies_settings_on_first_run(self, tmp_path):
        """Bundled settings.yml must be written to SETTINGS_FILE on first call."""
        assert not mgr.SETTINGS_FILE.exists()
        mgr._ensure_settings()
        assert mgr.SETTINGS_FILE.exists()
        content = mgr.SETTINGS_FILE.read_text()
        # Must include the critical JSON format line
        assert "json" in content

    def test_does_not_overwrite_existing(self, tmp_path):
        """_ensure_settings must never overwrite a file the user already has."""
        mgr.SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        mgr.SETTINGS_FILE.write_text("user-custom-content")
        mgr._ensure_settings()
        assert mgr.SETTINGS_FILE.read_text() == "user-custom-content"

    def test_creates_parent_directory(self, tmp_path):
        assert not mgr.SEARXNG_DIR.exists()
        mgr._ensure_settings()
        assert mgr.SEARXNG_DIR.exists()

    def test_get_settings_path_returns_path(self):
        path = mgr.get_settings_path()
        assert path == mgr.SETTINGS_FILE
        assert mgr.SETTINGS_FILE.exists()


# ---------------------------------------------------------------------------
# PID file management
# ---------------------------------------------------------------------------


class TestPidManagement:
    def test_write_and_read_pid(self):
        mgr._write_pid(12345)
        assert mgr._read_pid() == 12345

    def test_read_missing_pid_returns_none(self):
        assert mgr._read_pid() is None

    def test_read_invalid_content_returns_none(self, tmp_path):
        mgr.SEARXNG_PID_FILE.write_text("not-a-number")
        assert mgr._read_pid() is None

    def test_clear_pid_removes_file(self):
        mgr._write_pid(99)
        mgr._clear_pid()
        assert not mgr.SEARXNG_PID_FILE.exists()

    def test_clear_pid_missing_file_no_error(self):
        # Should not raise even if file doesn't exist
        mgr._clear_pid()


# ---------------------------------------------------------------------------
# is_running
# ---------------------------------------------------------------------------


class TestIsRunning:
    def test_no_pid_file_returns_false(self):
        assert mgr.is_running() is False

    def test_dead_process_clears_pid_returns_false(self):
        mgr._write_pid(999999)  # very unlikely to be a real PID
        with patch("os.kill", side_effect=ProcessLookupError):
            result = mgr.is_running()
        assert result is False
        assert not mgr.SEARXNG_PID_FILE.exists()

    def test_alive_process_unhealthy_returns_false(self):
        mgr._write_pid(12345)
        with (
            patch("os.kill"),  # process alive
            patch("httpx.get", side_effect=Exception("Connection refused")),
        ):
            result = mgr.is_running()
        assert result is False
        assert not mgr.SEARXNG_PID_FILE.exists()

    def test_alive_and_healthy_returns_true(self):
        mgr._write_pid(12345)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with (
            patch("os.kill"),
            patch("httpx.get", return_value=mock_resp),
        ):
            result = mgr.is_running()
        assert result is True

    def test_permission_error_treated_as_alive(self):
        """PermissionError on kill(pid, 0) means the process exists but isn't owned."""
        mgr._write_pid(12345)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with (
            patch("os.kill", side_effect=PermissionError),
            patch("httpx.get", return_value=mock_resp),
        ):
            result = mgr.is_running()
        assert result is True


# ---------------------------------------------------------------------------
# start_daemon
# ---------------------------------------------------------------------------


class TestStartDaemon:
    def test_successful_start_returns_true(self, tmp_path):
        """start_daemon returns True when SearXNG health check passes."""
        mock_proc = MagicMock()
        mock_proc.pid = 42
        mock_proc.poll.return_value = None  # process still alive

        with (
            patch("importlib.import_module"),  # mock importability check
            patch.object(mgr, "_spawn_subprocess", return_value=mock_proc),
            patch.object(mgr, "_health_check", return_value=True),
            patch.object(mgr, "_ensure_settings"),
        ):
            result = mgr.start_daemon()

        assert result is True
        assert mgr._read_pid() == 42

    def test_process_dies_immediately_returns_false(self, tmp_path):
        mock_proc = MagicMock()
        mock_proc.pid = 43
        mock_proc.poll.return_value = 1  # process exited immediately

        with (
            patch("importlib.import_module"),
            patch.object(mgr, "_spawn_subprocess", return_value=mock_proc),
            patch.object(mgr, "_health_check", return_value=False),
            patch.object(mgr, "_ensure_settings"),
        ):
            result = mgr.start_daemon(startup_timeout=1.0)

        assert result is False
        assert not mgr.SEARXNG_PID_FILE.exists()

    def test_timeout_returns_false(self, tmp_path):
        mock_proc = MagicMock()
        mock_proc.pid = 44
        mock_proc.poll.return_value = None  # alive but never healthy

        with (
            patch("importlib.import_module"),
            patch.object(mgr, "_spawn_subprocess", return_value=mock_proc),
            patch.object(mgr, "_health_check", return_value=False),
            patch.object(mgr, "_ensure_settings"),
            patch("time.sleep"),  # don't actually sleep in tests
            patch("time.monotonic", side_effect=[0.0, 0.6, 1.2, 1.5]),  # exceeds timeout=1.0
        ):
            result = mgr.start_daemon(startup_timeout=1.0)

        assert result is False

    def test_searx_not_importable_returns_false(self, tmp_path):
        """Bug 1 fix: missing searx module must return False, not raise RuntimeError."""
        with (
            patch("importlib.import_module", side_effect=ImportError("No module named searx")),
            patch.object(mgr, "_ensure_settings"),
        ):
            result = mgr.start_daemon()
        assert result is False

    def test_writes_pid_file(self, tmp_path):
        mock_proc = MagicMock()
        mock_proc.pid = 55
        mock_proc.poll.return_value = None

        with (
            patch("importlib.import_module"),
            patch.object(mgr, "_spawn_subprocess", return_value=mock_proc),
            patch.object(mgr, "_health_check", return_value=True),
            patch.object(mgr, "_ensure_settings"),
        ):
            mgr.start_daemon()

        assert mgr._read_pid() == 55


# ---------------------------------------------------------------------------
# stop_daemon
# ---------------------------------------------------------------------------


class TestStopDaemon:
    def test_stop_when_not_running_returns_false(self):
        assert mgr.stop_daemon() is False

    def test_stop_dead_process_returns_false(self):
        mgr._write_pid(999999)
        with patch("os.kill", side_effect=ProcessLookupError):
            result = mgr.stop_daemon()
        assert result is False
        assert not mgr.SEARXNG_PID_FILE.exists()

    def test_stop_running_process_returns_true(self):
        mgr._write_pid(12345)
        kill_calls = []

        def fake_kill(pid, sig):
            kill_calls.append((pid, sig))
            if sig == signal.SIGTERM:
                # Simulate process dying after SIGTERM
                with patch("tinirag.core.searxng_manager._process_alive", return_value=False):
                    pass

        with (
            patch("os.kill", side_effect=fake_kill),
            patch("tinirag.core.searxng_manager._process_alive", return_value=False),
            patch("time.sleep"),
        ):
            result = mgr.stop_daemon()

        assert result is True
        assert not mgr.SEARXNG_PID_FILE.exists()
        # SIGTERM must have been sent
        assert any(sig == signal.SIGTERM for _, sig in kill_calls)

    def test_clears_pid_file_after_stop(self):
        mgr._write_pid(12345)
        with (
            patch("os.kill"),
            patch("tinirag.core.searxng_manager._process_alive", return_value=False),
            patch("time.sleep"),
        ):
            mgr.stop_daemon()
        assert not mgr.SEARXNG_PID_FILE.exists()


# ---------------------------------------------------------------------------
# ensure_running
# ---------------------------------------------------------------------------


class TestEnsureRunning:
    def test_already_running_returns_true_without_start(self):
        """If is_running() returns True, start_daemon must NOT be called."""
        with (
            patch.object(mgr, "is_running", return_value=True),
            patch.object(mgr, "start_daemon") as mock_start,
        ):
            result = mgr.ensure_running()
        assert result is True
        mock_start.assert_not_called()

    def test_not_running_calls_start_daemon(self):
        with (
            patch.object(mgr, "is_running", return_value=False),
            patch.object(mgr, "start_daemon", return_value=True) as mock_start,
        ):
            result = mgr.ensure_running()
        assert result is True
        mock_start.assert_called_once()

    def test_start_failure_propagates_false(self):
        with (
            patch.object(mgr, "is_running", return_value=False),
            patch.object(mgr, "start_daemon", return_value=False),
        ):
            result = mgr.ensure_running()
        assert result is False
