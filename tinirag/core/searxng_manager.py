"""SearXNG daemon lifecycle manager.

Starts SearXNG as a background subprocess on first use and keeps it running
across tinirag invocations. Users never need to install or manage Docker.
"""

from __future__ import annotations

import importlib.resources
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx

from tinirag.config import SEARXNG_DIR, SEARXNG_LOG_FILE, SEARXNG_PID_FILE

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

SETTINGS_FILE = SEARXNG_DIR / "settings.yml"
DEFAULT_PORT = 18888
STARTUP_TIMEOUT_SEC = 30


# ---------------------------------------------------------------------------
# Settings management
# ---------------------------------------------------------------------------


def get_settings_path() -> Path:
    """Return path to user's settings.yml, copying from package data on first run."""
    _ensure_settings()
    return SETTINGS_FILE


def _ensure_settings() -> None:
    """Copy bundled settings.yml to ~/.tinirag/searxng/ on first run only.

    Never overwrites an existing file — the user's edits are preserved.
    """
    if SETTINGS_FILE.exists():
        return
    SEARXNG_DIR.mkdir(parents=True, exist_ok=True)
    bundled = importlib.resources.files("tinirag.data").joinpath("settings.yml")
    SETTINGS_FILE.write_text(bundled.read_text(encoding="utf-8"), encoding="utf-8")


# ---------------------------------------------------------------------------
# PID file management
# ---------------------------------------------------------------------------


def _read_pid() -> int | None:
    """Read PID from pid file. Returns None if missing or invalid."""
    try:
        return int(SEARXNG_PID_FILE.read_text().strip())
    except Exception:
        return None


def _write_pid(pid: int) -> None:
    """Write PID to pid file."""
    SEARXNG_PID_FILE.write_text(str(pid))


def _clear_pid() -> None:
    """Remove pid file if it exists."""
    SEARXNG_PID_FILE.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Runtime checks
# ---------------------------------------------------------------------------


def _process_alive(pid: int) -> bool:
    """Return True if the process with this PID is alive."""
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists but we don't own it


def _health_check(port: int) -> bool:
    """Return True if SearXNG responds on /healthz."""
    try:
        r = httpx.get(f"http://127.0.0.1:{port}/healthz", timeout=1.0)
        return r.status_code == 200
    except Exception:
        return False


def _spawn_subprocess(env: dict) -> subprocess.Popen:  # type: ignore[type-arg]
    """Launch searx.webapp as a detached background process. Extracted for testability."""
    SEARXNG_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    log_fd = open(SEARXNG_LOG_FILE, "a")  # noqa: SIM115 — kept open for subprocess lifetime
    return subprocess.Popen(
        [sys.executable, "-m", "searx.webapp"],
        env=env,
        stdout=log_fd,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def is_running(port: int = DEFAULT_PORT) -> bool:
    """Return True if a managed SearXNG process is alive and healthy."""
    pid = _read_pid()
    if pid is None:
        return False
    if not _process_alive(pid):
        _clear_pid()
        return False
    if not _health_check(port):
        _clear_pid()
        return False
    return True


# ---------------------------------------------------------------------------
# Daemon start / stop
# ---------------------------------------------------------------------------


def start_daemon(
    port: int = DEFAULT_PORT,
    startup_timeout: float = STARTUP_TIMEOUT_SEC,
) -> bool:
    """Start SearXNG as a detached background subprocess.

    Uses sys.executable so the same pip env that tinirag runs in is used,
    guaranteeing searx.webapp is importable regardless of venv / pipx setup.

    Returns True if SearXNG is healthy within startup_timeout seconds.
    """
    _ensure_settings()

    # Verify searx is importable before attempting to spawn
    try:
        import importlib

        importlib.import_module("searx.webapp")
    except ImportError as exc:
        raise RuntimeError(
            "searx.webapp not importable. Ensure searxng is installed: pip install searxng"
        ) from exc

    env = os.environ.copy()
    env["SEARXNG_SETTINGS_PATH"] = str(SETTINGS_FILE)

    try:
        proc = _spawn_subprocess(env)
    except Exception as exc:
        raise RuntimeError(f"Failed to start SearXNG subprocess: {exc}") from exc

    _write_pid(proc.pid)

    # Poll health with 0.5s interval — SearXNG typically ready in 3-5s
    deadline = time.monotonic() + startup_timeout
    while time.monotonic() < deadline:
        # Fast-fail if the process died immediately
        if proc.poll() is not None:
            _clear_pid()
            return False
        if _health_check(port):
            return True
        time.sleep(0.5)

    return False


def stop_daemon() -> bool:
    """Stop the managed SearXNG process.

    Returns True if a process was stopped, False if none was running.
    """
    pid = _read_pid()
    if pid is None:
        return False

    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        _clear_pid()
        return False

    # Wait up to 5s for graceful shutdown
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if not _process_alive(pid):
            break
        time.sleep(0.2)
    else:
        # Force-kill if still alive
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    _clear_pid()
    return True


# ---------------------------------------------------------------------------
# High-level entry point (called from engine.startup_check)
# ---------------------------------------------------------------------------


def ensure_running(
    port: int = DEFAULT_PORT,
    startup_timeout: float = STARTUP_TIMEOUT_SEC,
) -> bool:
    """Ensure SearXNG is running. Start it if not. Returns True on success.

    Idempotent — safe to call on every query. The PID+health check fast-path
    adds <5ms when already running.
    """
    if is_running(port):
        return True
    return start_daemon(port, startup_timeout)
