"""TiniRAG CLI entry point."""

from __future__ import annotations

import asyncio
import json
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from tinirag.config import GUARDRAIL_LOG, load_config
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


def _run(coro: object) -> object:
    """Run a coroutine from a sync Typer command."""
    return asyncio.run(coro)  # type: ignore[arg-type]


def _handle_result(result: QueryResult, cfg_output: object, verify: bool) -> None:
    """Print warnings, sources, and (optionally) verify claims."""
    for w in result.warnings:
        print_warning(w)
    if getattr(cfg_output, "show_sources", True) and result.sources:
        print_sources(result.sources)


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
        if not no_search:
            await startup_check(cfg)
        try:
            result = await run_query(
                query,
                cfg,
                no_search=no_search,
                verify=verify,
                cache=cache,
                history=history,
            )
        except ValueError as exc:
            print_error(str(exc))
            raise typer.Exit(1) from exc

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
        if not no_search:
            await startup_check(cfg)

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
    else:
        console.print("Use [bold]tinirag logs --rails[/bold] to view guard rail events.")
