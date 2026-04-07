# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run all tests
uv run pytest -q

# Run a single test file
uv run pytest tests/test_engine.py -q

# Run a single test by name
uv run pytest -k "test_startup_check" -q

# Lint
uv run ruff check .

# Install deps
uv sync
```

Always run `uv run pytest -q` before making changes to verify the baseline is green.

## Architecture

TiniRAG is a privacy-first RAG CLI that combines SearXNG web search with a local LLM (via any OpenAI-compatible endpoint). No cloud APIs, no LangChain/LlamaIndex.

**Pipeline flow:**
```
User query
  → Tier 1 regex optimizer (<1ms, handles ~80%)
  → Tier 2 LLM optimizer (only if result > 8 words)
  → SearXNG JSON search + asyncio.gather(URL fetches)  ← parallel
  → Context builder (chunk, deduplicate, token-budget)
  → Layer 2 grounded RAG prompt → local LLM (streaming)
  → Layer 3 fallback if zero results
```

**Key modules (expected structure):**
- `tinirag/cli.py` — Typer entry point; `tinirag "query"` (direct), `tinirag chat` (REPL), `tinirag sessions`
- `tinirag/core/engine.py` — `startup_check()`, `is_ollama_running`, `check_model_available`, `pull_model` as module-level (not local) so tests can patch via `patch("tinirag.core.engine.X")`
- `tinirag/core/renderer.py` — `stream_response_live` is async (`async for token` inside sync `with Live(...)`)
- `tinirag/config.py` — `auto_select_backend()` picks brave → tavily → ddg based on API keys set; reads `~/.tinirag/config.toml`

**Config hierarchy:** CLI flags → env vars → `~/.tinirag/config.toml` → built-in defaults

## Critical Bugs to Avoid

**BUG-01 — SearXNG JSON disabled by default:** Docker image returns `200 OK` with HTML when `format=json` not enabled. Always validate `content-type: application/json` before calling `.json()`. Requires `search.formats: [html, json]` in `settings.yml`.

**BUG-02 — tiktoken KeyError on local model names:** `tiktoken.encoding_for_model("llama3")` raises `KeyError`. Use the safe wrapper with `cl100k_base` fallback + 10–15% token budget safety margin.

**BUG-03 — trafilatura returns None silently:** Always None-check `trafilatura.extract()` and apply a 100-char minimum content threshold. Paywall/JS pages return `None`, not `""`.

**BUG-05 — Injection regex not compiled:** Pre-compile the injection detection regex with `(?i)` flag at module load, not per-call.

**BUG-06 — Context in system message breaks Mistral:** Always inject retrieved context into the **user** message, not the system message. Use `---` delimiters.

**BUG-07 — Multi-word phrase matching fails cross-sentence:** Flatten phrases to individual words before substring matching in relevance checks.

**BUG-08 — Cache key collision risk:** Use `sha256(normalized_keywords)[:32]` (128-bit), not 16-char.

## Design Constraints

- **No LangChain/LlamaIndex** — orchestration stays in-house for auditability and stability.
- **No embedding models or vector DB** — v1 uses keyword search only (SearXNG).
- **Context goes in user message** — never system message (BUG-06).
- **GR-Q3 and GR-G1 are hardcoded non-negotiables** — injection detection and abstention instruction cannot be disabled by config.
- **Cache keys:** normalize by sorted words + stop-word removal before hashing so rephrased queries deduplicate.
- **Token counting:** always use the `count_tokens()` safe wrapper; `tiktoken` only knows OpenAI model names.
- **Parallel fetching:** `asyncio.gather()` for all URL fetches simultaneously (OPT-01). One persistent `httpx.AsyncClient` per pipeline run, not per request (OPT-05).
- **Per-fetch timeout:** 2.5s hard cap; gracefully degrade to snippet fallback on failure (OPT-02).
- **Char-truncate before token counting:** truncate to `~1600 chars` per source before calling `count_tokens()` — saves ~60% token-counting overhead (OPT-08).

## LLM Endpoint Auto-Detection

Probe ports in order: Ollama (11434) → vLLM/transformers (8000) → LM Studio (1234) → llama.cpp (8080). Configured via `TINIRAG_ENDPOINT` env var or `llm.endpoint` in config.toml.

## SearXNG Setup

```bash
# Minimal Docker setup
docker run -d --name searxng -p 8888:8080 searxng/searxng:latest
# settings.yml must include: search.formats: [html, json]
```

API call: `GET /search?q={keywords}&format=json&categories=general&pageno=1&time_range=year`
(`time_range=year` reduces SearXNG aggregation latency 20–40%.)

## Python Version Notes

- Minimum: Python 3.10+
- `tomllib` is stdlib only on 3.11+; use `tomli` on 3.10 (conditional import)
- `tomllib` is **read-only** — use `tomli_w` for all config writes
