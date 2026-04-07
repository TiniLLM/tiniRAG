"""Terminal output: streaming LLM responses and sources block."""

from __future__ import annotations

from datetime import date
from typing import AsyncIterator

from rich.console import Console
from rich.markup import escape
from rich.rule import Rule
from rich.text import Text

console = Console()


async def stream_response_live(token_stream: AsyncIterator[str]) -> str:
    """Stream tokens to stdout as they arrive.

    Uses async iteration directly — avoids the broken _AsyncToSyncIterator
    pattern that silently fails when an event loop is already running.
    flush=True equivalent is implicit in Rich's Console live output.
    """
    full_response: list[str] = []
    async for token in token_stream:
        if token:
            print(token, end="", flush=True)  # flush=True: prevents buffering (GP 4.4)
            full_response.append(token)
    print()  # final newline
    return "".join(full_response)


def print_sources(sources: list[dict]) -> None:
    """Print the provenance/sources block at the end of every response (GP 5.3)."""
    if not sources:
        return
    today = date.today().isoformat()
    console.print()
    console.print(Rule(style="dim"))
    console.print(Text("Sources", style="bold dim"))
    for s in sources:
        num = s.get("_source_num", "?")
        url = escape(
            s.get("url", "unknown")
        )  # escape brackets so IPv6/unusual URLs don't break markup
        console.print(f"  [{num}] {url}  (retrieved {today})", style="dim")
    console.print(Rule(style="dim"))


def print_warning(msg: str) -> None:
    console.print(f"[bold yellow][!][/bold yellow] {msg}")


def print_error(msg: str) -> None:
    console.print(f"[bold red][✗][/bold red] {msg}")


def print_info(msg: str) -> None:
    console.print(f"[bold blue][→][/bold blue] {msg}")


async def _collect_stream(response: object) -> AsyncIterator[str]:
    """Convert an openai streaming response into an async string token iterator."""
    async for chunk in response:  # type: ignore[union-attr]
        # Some endpoints emit final "usage" chunks with an empty choices list
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
