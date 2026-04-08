"""TiniRAG CLI entry point."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from tinirag.config import GUARDRAIL_LOG, SEARXNG_LOG_FILE, load_config, save_config
from tinirag.core.cache import make_cache
from tinirag.core.engine import QueryResult, run_query, startup_check
from tinirag.core.renderer import print_error, print_info, print_sources, print_warning
from tinirag.core.session import (
    list_sessions,
    load_session,
    new_session_id,
    save_session,
)

app = typer.Typer(
    name="tinirag",
    help="Privacy-first RAG CLI — local LLM + SearXNG, zero cloud.",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()

_DEBUG = os.getenv("TINIRAG_DEBUG") == "1"


def _run(coro: object) -> object:
    """Run a coroutine from a sync Typer command."""
    return asyncio.run(coro)  # type: ignore[arg-type]


def _handle_result(result: QueryResult, cfg_output: object, verify: bool) -> None:
    """Print warnings, sources, and (optionally) verify claims."""
    for w in result.warnings:
        print_warning(w)
    if getattr(cfg_output, "show_sources", True) and result.sources:
        print_sources(result.sources)


def _handle_llm_error(exc: Exception, endpoint: str, model: str | None) -> None:
    """Print a clean error for known LLM failure modes. Re-raise unexpected errors."""
    import httpx as _httpx

    try:
        import openai as _openai

        _openai_errors = (_openai.NotFoundError, _openai.APIConnectionError)
    except ImportError:
        _openai_errors = ()  # type: ignore[assignment]

    if _DEBUG:
        raise exc

    exc_str = str(exc).lower()

    if _openai_errors and isinstance(exc, _openai.NotFoundError) and "model" in exc_str:
        name = model or "unknown"
        print_error(
            f"Model '{name}' not found in Ollama.\n"
            f"    Run `ollama list` to see installed models.\n"
            f"    Run `ollama pull {name}` to install it."
        )
    elif _openai_errors and isinstance(exc, _openai.APIConnectionError):
        print_error(f"Cannot reach Ollama at {endpoint}.\n    Is Ollama running? Try: ollama serve")
    elif isinstance(exc, _httpx.ConnectError):
        print_error(f"Cannot connect to {endpoint}.\n    Is Ollama running? Try: ollama serve")
    else:
        raise exc  # unexpected — show traceback


# ---------------------------------------------------------------------------
# Default command: tinirag "query"
# ---------------------------------------------------------------------------


@app.callback(invoke_without_command=True)
def _default(
    ctx: typer.Context,
    query: Optional[str] = typer.Argument(None, help="Your question or search query."),
    no_search: bool = typer.Option(False, "--no-search", help="Skip web search; query LLM only."),
    verify: bool = typer.Option(False, "--verify", help="Enable hallucination smoke test (slow)."),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Override LLM model name."),
    endpoint: Optional[str] = typer.Option(None, "--endpoint", help="Override LLM endpoint URL."),
    history: bool = typer.Option(False, "--history", help="Append query to history.jsonl."),
) -> None:
    """Run a single RAG query and print the grounded answer."""
    if ctx.invoked_subcommand is not None:
        return
    if not query:
        console.print(ctx.get_help())
        raise typer.Exit()

    cfg = load_config()
    if model:
        cfg.llm.model = model
    if endpoint:
        cfg.llm.endpoint = endpoint

    cache = make_cache(cfg.cache.backend, cfg.cache.ttl_minutes) if cfg.cache.enabled else None

    async def _run_it() -> None:
        try:
            # Always run startup_check — passes no_search so SearXNG is skipped when disabled
            await startup_check(cfg, no_search=no_search)

            result = await run_query(
                query,
                cfg,
                no_search=no_search,
                verify=verify,
                cache=cache,
                history=history,
            )
        except SystemExit:
            raise
        except ValueError as exc:
            print_error(str(exc))
            raise typer.Exit(1) from exc
        except Exception as exc:
            _handle_llm_error(exc, cfg.llm.endpoint, cfg.llm.model)
            raise typer.Exit(1)

        if cfg.output.show_keywords and result.keywords:
            print_info(f"Keywords: {result.keywords}")
        _handle_result(result, cfg.output, verify)

    _run(_run_it())


# ---------------------------------------------------------------------------
# chat — interactive REPL
# ---------------------------------------------------------------------------


@app.command()
def chat(
    no_search: bool = typer.Option(False, "--no-search", help="Skip web search."),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Override model."),
    session_id: Optional[str] = typer.Option(None, "--session", help="Resume a saved session."),
) -> None:
    """Start an interactive chat REPL with session memory."""
    cfg = load_config()
    if model:
        cfg.llm.model = model

    cache = make_cache(cfg.cache.backend, cfg.cache.ttl_minutes) if cfg.cache.enabled else None
    sid = session_id or new_session_id()
    messages: list[dict] = []
    title = ""

    if session_id:
        saved = load_session(session_id)
        if saved:
            messages = saved.get("messages", [])
            title = saved.get("title", "")
            print_info(f"Resumed session {sid}: {title}")
        else:
            print_warning(f"Session '{session_id}' not found. Starting new session.")

    print_info(f"TiniRAG chat — session {sid}. Type 'exit' or Ctrl-C to quit.")

    async def _loop() -> None:
        nonlocal title
        try:
            await startup_check(cfg, no_search=no_search)
        except SystemExit:
            raise
        except Exception as exc:
            _handle_llm_error(exc, cfg.llm.endpoint, cfg.llm.model)
            raise typer.Exit(1)

        while True:
            try:
                raw = input("\n[You] ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if raw.lower() in ("exit", "quit", ":q"):
                break
            if not raw:
                continue

            try:
                result = await run_query(
                    raw,
                    cfg,
                    no_search=no_search,
                    cache=cache,
                )
            except ValueError as exc:
                print_error(str(exc))
                continue
            except Exception as exc:
                _handle_llm_error(exc, cfg.llm.endpoint, cfg.llm.model)
                continue

            messages.append({"role": "user", "content": raw})
            messages.append({"role": "assistant", "content": result.response})

            if not title:
                title = raw[:60]

            for w in result.warnings:
                print_warning(w)
            if cfg.output.show_sources and result.sources:
                print_sources(result.sources)

        if messages:
            path = save_session(sid, messages, title)
            print_info(f"Session saved: {path}")

    _run(_loop())


# ---------------------------------------------------------------------------
# sessions — list saved sessions
# ---------------------------------------------------------------------------


@app.command()
def sessions() -> None:
    """List all saved chat sessions."""
    all_sessions = list_sessions()
    if not all_sessions:
        print_info("No saved sessions found.")
        return

    table = Table(title="Saved Sessions", show_lines=True)
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Title")
    table.add_column("Created", style="dim")
    table.add_column("Messages", justify="right", style="dim")

    for s in all_sessions:
        table.add_row(
            s["id"],
            s["title"],
            s["created"][:19].replace("T", " ") if s["created"] else "",
            str(s["message_count"]),
        )
    console.print(table)


# ---------------------------------------------------------------------------
# logs — view guard rail events
# ---------------------------------------------------------------------------


@app.command()
def logs(
    rails: bool = typer.Option(False, "--rails", help="Show guard rail activation log."),
    searxng: bool = typer.Option(False, "--searxng", help="Show SearXNG startup log."),
    n: int = typer.Option(20, "--last", "-n", help="Number of recent entries to show."),
) -> None:
    """View TiniRAG logs."""
    if rails:
        if not GUARDRAIL_LOG.exists():
            print_info("No guard rail events logged yet.")
            return
        lines = GUARDRAIL_LOG.read_text().strip().splitlines()
        recent = lines[-n:]
        table = Table(title="Guard Rail Activations", show_lines=True)
        table.add_column("Time", style="dim")
        table.add_column("Rail", style="cyan")
        table.add_column("Trigger")
        table.add_column("Query", style="dim")
        for line in recent:
            try:
                entry = json.loads(line)
                table.add_row(
                    entry.get("ts", "")[:19].replace("T", " "),
                    entry.get("rail", ""),
                    entry.get("trigger", ""),
                    entry.get("query", ""),
                )
            except Exception:
                continue
        console.print(table)
    elif searxng:
        if not SEARXNG_LOG_FILE.exists():
            print_info("No SearXNG log found. Run a query first to start SearXNG.")
            return
        lines = SEARXNG_LOG_FILE.read_text().splitlines()
        for line in lines[-n:]:
            console.print(line, highlight=False)
    else:
        console.print(
            "Use [bold]tinirag logs --rails[/bold] for guard rail events\n"
            "or  [bold]tinirag logs --searxng[/bold] for SearXNG startup logs."
        )


# ---------------------------------------------------------------------------
# stop — stop background SearXNG process
# ---------------------------------------------------------------------------


@app.command()
def stop() -> None:
    """Stop the background SearXNG process managed by TiniRAG."""
    from tinirag.core.searxng_manager import is_running, stop_daemon

    if not is_running():
        print_info("SearXNG is not running.")
        return
    stopped = stop_daemon()
    if stopped:
        print_info("SearXNG stopped.")
    else:
        print_warning("Could not stop SearXNG — process not found or permission denied.")


# ---------------------------------------------------------------------------
# status — show component health
# ---------------------------------------------------------------------------


@app.command()
def status() -> None:
    """Show status of SearXNG and Ollama backends."""
    from tinirag.core.engine import _endpoint_base, check_model_available, is_ollama_running
    from tinirag.core.searxng_manager import is_running

    cfg = load_config()

    table = Table(title="TiniRAG Status", show_lines=True)
    table.add_column("Component", style="cyan")
    table.add_column("Status")
    table.add_column("Detail", style="dim")

    # SearXNG status
    if cfg.search.managed_searxng:
        if is_running(cfg.search.searxng_port):
            table.add_row("SearXNG", "[green]running[/green]", cfg.search.searxng_url)
        else:
            table.add_row(
                "SearXNG",
                "[red]stopped[/red]",
                "run `tinirag <query>` to auto-start",
            )
    else:
        ok = asyncio.run(
            __import__("tinirag.core.search", fromlist=["check_searxng"]).check_searxng(
                cfg.search.searxng_url
            )
        )
        label = "[green]reachable[/green]" if ok else "[red]unreachable[/red]"
        table.add_row("SearXNG", label, f"user-managed at {cfg.search.searxng_url}")

    # Ollama status
    endpoint_base = _endpoint_base(cfg.llm.endpoint)
    if is_ollama_running(endpoint_base):
        # Resolve model for display: use configured value or auto-detect
        display_model = cfg.llm.model
        if display_model is None:
            from tinirag.core.model_detect import detect_available_model

            display_model = detect_available_model(endpoint_base)

        if display_model:
            model_ok = check_model_available(display_model, endpoint_base)
            detail = (
                f"model '{display_model}' ready"
                if model_ok
                else f"model '{display_model}' not pulled — run `ollama pull {display_model}`"
            )
        else:
            detail = "no models installed — run `ollama pull llama3.2:3b`"
        table.add_row("Ollama", "[green]running[/green]", detail)
    else:
        table.add_row("Ollama", "[red]not running[/red]", "run: ollama serve")

    console.print(table)


# ---------------------------------------------------------------------------
# setup — guided first-run wizard
# ---------------------------------------------------------------------------


@app.command()
def setup(
    with_search: bool = typer.Option(False, "--with-search", help="Also configure SearXNG search."),
) -> None:
    """Guided first-run setup: verify install, detect model, optionally configure search."""
    from tinirag.core.engine import _endpoint_base, is_ollama_running
    from tinirag.core.model_detect import detect_available_model

    cfg = load_config()
    console.print("[bold blue]TiniRAG Setup[/bold blue]\n")

    # 1. Verify Python version
    if sys.version_info < (3, 11):
        print_error(
            f"Python 3.11+ required. You have {sys.version_info.major}.{sys.version_info.minor}"
        )
        raise typer.Exit(1)
    print_info(f"Python {sys.version_info.major}.{sys.version_info.minor}: OK")

    # 2. Check Ollama reachability
    endpoint_base = _endpoint_base(cfg.llm.endpoint)
    if not is_ollama_running(endpoint_base):
        print_error(
            "Ollama not running.\n    Install: https://ollama.com\n    Then run: ollama serve"
        )
        raise typer.Exit(1)
    print_info("Ollama: running")

    # 3. List installed models / auto-detect best one
    detected = detect_available_model(endpoint_base)
    if detected:
        print_info(f"Auto-detects your installed Ollama models — using: {detected}")
        cfg.llm.model = detected
    else:
        # No models found — offer to pull one
        answer = typer.prompt(
            "\nNo models found. Pull recommended model llama3.2:3b? [Y/n]", default="Y"
        )
        if answer.strip().upper() in ("Y", ""):
            print_info("Pulling llama3.2:3b (this may take a few minutes)...")
            result = subprocess.run(["ollama", "pull", "llama3.2:3b"])
            if result.returncode == 0:
                print_info("Model llama3.2:3b ready.")
                cfg.llm.model = "llama3.2:3b"
            else:
                print_error("Pull failed. Try manually: ollama pull llama3.2:3b")
                raise typer.Exit(1)
        else:
            print_warning("No model configured. Set one with: tinirag -m <model-name> ...")

    # 4. Save config with detected model
    if cfg.llm.model:
        save_config(cfg)
        print_info(f"Config saved. Model: {cfg.llm.model}")

    # 5. SearXNG (only with --with-search flag)
    if with_search:
        try:
            import importlib

            importlib.import_module("searx.webapp")
            from tinirag.core.searxng_manager import get_settings_path

            settings_path = get_settings_path()
            print_info(f"SearXNG: found — settings at {settings_path}")
        except ImportError:
            print_warning(
                "searx package not installed. To enable search:\n"
                "    pipx inject tinirag searxng\n"
                "    (or: pip install searxng  if not using pipx)"
            )

    console.print('\n[bold green]Setup complete![/bold green] Run: tinirag "your question"')
