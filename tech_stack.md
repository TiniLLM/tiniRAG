# TiniRAG — Tech Stack

## Architecture Philosophy

TiniRAG is a **zero-dependency-on-cloud**, privacy-first CLI tool.
Every component runs locally or on user-controlled infrastructure.
The stack is chosen for:
- Model-agnostic LLM integration (any OpenAI-compatible endpoint)
- Minimal install surface (single `pip install tinirag`)
- Low resource footprint (runs on CPU-only machines)
- Compatibility with every HuggingFace-downloadable model

---

## Component Map

```
┌─────────────────────────────────────────────────────────┐
│                     TiniRAG CLI                         │
│                   (Python, Typer)                       │
└──────────────┬──────────────────────────┬───────────────┘
               │                          │
       ┌───────▼────────┐        ┌────────▼────────┐
       │ Query Optimizer│        │  Result Cache   │
       │ (regex + LLM)  │        │  (SQLite / dict)│
       └───────┬────────┘        └─────────────────┘
               │
       ┌───────▼────────┐
       │    SearXNG     │  ◄── Self-hosted (Docker)
       │  (JSON API)    │
       └───────┬────────┘
               │
       ┌───────▼────────┐
       │  Web Fetcher   │  ◄── trafilatura + httpx
       │ (optional, on- │
       │  demand fetch) │
       └───────┬────────┘
               │
       ┌───────▼────────┐
       │ Context Builder│  ◄── chunker + deduplicator
       └───────┬────────┘
               │
       ┌───────▼────────────────────────────────────┐
       │           Local LLM Endpoint               │
       │      POST /v1/chat/completions             │
       │                                            │
       │  Ollama │ vLLM │ llama.cpp │ LM Studio    │
       │  transformers serve │ TGI │ LMDeploy      │
       └────────────────────────────────────────────┘
```

---

## Core Stack

### CLI Framework
| Tool | Version | Purpose |
|------|---------|---------|
| **Typer** | `>=0.12` | CLI argument parsing, subcommands, help text |
| **Rich** | `>=13.0` | Coloured terminal output, progress bars, tables |

**Why Typer:** Built on Click, supports auto-generated help, type hints,
and shell completion. Minimal boilerplate for a clean CLI.

---

### Search Layer

| Tool | Purpose |
|------|---------|
| **SearXNG** | Self-hosted meta-search engine. Aggregates Google, Bing, Brave, DuckDuckGo, Wikipedia, GitHub |
| **Docker / Podman** | Runtime for SearXNG container |
| **httpx** | Async HTTP client for SearXNG JSON API calls |

**SearXNG Docker setup (minimal):**
```bash
docker run -d \
  --name searxng \
  -p 8888:8080 \
  -e BASE_URL=http://localhost:8080 \
  searxng/searxng:latest
```

> ⚠️ **BUG-01 — CRITICAL:** The SearXNG Docker image ships with JSON format
> **disabled by default**. The API returns `200 OK` with an HTML body when
> `format=json` is not enabled — it does NOT return a 4xx error. Your JSON
> parser will crash or silently pass raw HTML as LLM context. This is the
> #1 reported integration failure in the searxng/searxng GitHub issues.
>
> **Required fix — create `/etc/searxng/settings.yml` before starting:**
> ```yaml
> search:
>   formats:
>     - html
>     - json   # ← THIS LINE IS MANDATORY. Without it, JSON is disabled.
> ```
>
> **Required fix — always validate the response content-type in code:**
> ```python
> response = await client.get(searxng_url, params=params)
> content_type = response.headers.get("content-type", "")
> if "application/json" not in content_type:
>     raise RuntimeError(
>         "SearXNG returned HTML instead of JSON. "
>         "Add 'json' to search.formats in settings.yml and restart the container."
>     )
> data = response.json()
> ```

**SearXNG JSON API call:**
```
GET http://localhost:8888/search?q={keywords}&format=json&categories=general&pageno=1&time_range=year
```

> Note: Always include `pageno=1` explicitly and `time_range=year` for speed.
> `time_range=year` limits SearXNG's internal aggregation to recent results,
> reducing response time by 20–40% on Google/Bing engines. (OPT-07)

**Recommended SearXNG engine priority (settings.yml):**
```yaml
engines:
  - name: google
    weight: 3
  - name: brave
    weight: 2
  - name: bing
    weight: 1
  - name: duckduckgo
    weight: 1
  - name: github
    weight: 2     # for code/technical queries
  - name: arxiv
    weight: 2     # for research queries
```

---

### Web Fetcher (Optional, On-demand)

| Tool | Purpose |
|------|---------|
| **trafilatura** | Best-in-class HTML → clean text extraction. Strips nav, ads, footers. |
| **httpx** | Async fetch with timeout and redirect handling |
| **markdownify** | Convert extracted HTML to markdown (preserves structure for LLMs) |

**Why trafilatura over BeautifulSoup:** trafilatura is purpose-built for
article/content extraction and outperforms manual CSS selectors on
arbitrary web pages. No manual rule maintenance needed.

```python
import trafilatura

async def fetch_and_extract(client: httpx.AsyncClient, url: str) -> str | None:
    """
    Fetch a URL and extract clean text. Returns None on failure or thin content.
    BUG-03: trafilatura.extract() returns None (not "") on paywalled/JS pages.
    Always None-check before using the result downstream.
    """
    try:
        response = await asyncio.wait_for(
            client.get(url, follow_redirects=True),
            timeout=2.5  # OPT-02: short per-fetch timeout, fail fast
        )
        html = response.text
    except Exception:
        return None  # graceful degradation per GR-R4

    text = trafilatura.extract(html, include_links=False, include_tables=True)

    # BUG-03 + GR-R5: None check AND minimum content threshold in one guard
    if not text or len(text.strip()) < 100:
        return None  # paywall, redirect, or JavaScript-only page

    # OPT-08: char-truncate BEFORE token counting (instant, saves ~60% overhead)
    MAX_CHARS_PER_SOURCE = 1600  # ≈ 400 tokens at 4 chars/token
    return text[:MAX_CHARS_PER_SOURCE]
```

---

### LLM Integration Layer

| Tool | Purpose |
|------|---------|
| **openai** Python SDK | Universal client for all OpenAI-compatible endpoints |
| **huggingface_hub.InferenceClient** | Drop-in replacement for HF-hosted models |

**Universal endpoint pattern with parallel fetch (OPT-01, OPT-02, OPT-05):**
```python
import asyncio
from contextlib import asynccontextmanager
from openai import AsyncOpenAI

# OPT-05: one persistent httpx session per pipeline run, not per request
@asynccontextmanager
async def get_http_client():
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(connect=2.0, read=5.0, write=2.0, pool=8.0),
        limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        headers={"User-Agent": "TiniRAG/1.0"},
        follow_redirects=True,
    ) as client:
        yield client

# OPT-01: SearXNG search + all URL fetches fire in parallel
async def search_and_fetch(keywords: str, searxng_url: str) -> list[dict]:
    async with get_http_client() as client:
        # Step 1: SearXNG search (fast, ~300ms)
        resp = await client.get(
            f"{searxng_url}/search",
            params={"q": keywords, "format": "json",
                    "categories": "general", "pageno": 1, "time_range": "year"}
        )
        content_type = resp.headers.get("content-type", "")
        if "application/json" not in content_type:
            raise RuntimeError("SearXNG returned HTML. Enable JSON in settings.yml.")
        results = resp.json().get("results", [])[:5]

        # Step 2: fire ALL URL fetches simultaneously (OPT-01)
        htmls = await asyncio.gather(
            *[fetch_and_extract(client, r["url"]) for r in results]
        )
        return list(zip(results, htmls))

# LLM client — use AsyncOpenAI for non-blocking calls
llm_client = AsyncOpenAI(
    base_url=os.getenv("TINIRAG_ENDPOINT", "http://localhost:11434/v1"),
    api_key="not-needed",
)

# MISSING-02: fuzzy Ollama model name resolver (llama3 → llama3:latest)
async def resolve_model_name(requested: str) -> str:
    models_resp = await llm_client.models.list()
    available = [m.id for m in models_resp.data]
    if requested in available:
        return requested
    for m in available:
        if m.startswith(requested + ":"):
            return m
    raise ValueError(f"Model '{requested}' not found. Available: {available}")
```

**Supported runtimes (tested):**

| Runtime | Default Port | Model Source |
|---------|-------------|--------------|
| Ollama | 11434 | Ollama Hub (GGUF) |
| vLLM | 8000 | HuggingFace Hub |
| transformers serve | 8000 | HuggingFace Hub |
| llama.cpp server | 8080 | GGUF files |
| LM Studio | 1234 | GGUF files |
| HF TGI | 80/443 | HuggingFace Hub |
| LMDeploy | 8000 | HuggingFace Hub |

---

### Context & Chunking

| Tool | Purpose |
|------|---------|
| **tiktoken** | Token counting approximation (OpenAI models only — see BUG-02 fix below) |
| Built-in Python | Sentence-boundary splitting, deduplication |

**No LangChain/LlamaIndex dependency.** TiniRAG keeps orchestration logic
in-house for performance, simplicity, and auditability.
This avoids the LangChain version-churn problem common in community setups.

> ⚠️ **BUG-02 — CRITICAL:** `tiktoken.encoding_for_model()` raises `KeyError`
> for any non-OpenAI model name (`llama3`, `mistral`, `qwen2.5`, etc.). These
> are the primary models TiniRAG is designed for. Always use this safe wrapper:
>
> ```python
> import tiktoken
>
> def count_tokens(text: str, model_name: str) -> int:
>     """
>     Safe token counter. Falls back gracefully for all local/HF models.
>     tiktoken only knows OpenAI model names — local models always hit the except.
>     The cl100k_base fallback is accurate enough for context budget enforcement.
>     """
>     try:
>         enc = tiktoken.encoding_for_model(model_name)
>         return len(enc.encode(text))
>     except KeyError:
>         # tiktoken doesn't know llama3, mistral, qwen, etc.
>         try:
>             enc = tiktoken.get_encoding("cl100k_base")
>             return len(enc.encode(text))
>         except Exception:
>             # Fully offline or tiktoken broken: use character estimate
>             return len(text) // 4
> ```
>
> Use a **10–15% safety margin** on the token budget when using the
> cl100k_base fallback, since Llama3/Qwen tokenizers produce slightly
> different counts than OpenAI's tokenizer.

---

### Caching

| Mode | Backend | When to use |
|------|---------|-------------|
| In-memory | Python `dict` | Single session, fast, no persistence |
| Persistent | SQLite via `sqlite3` (stdlib) | Multi-session, cross-run cache |

Cache key: `sha256(normalized_keywords)[:32]`
TTL: 10 minutes (configurable via `--cache-ttl`)

> ⚠️ **BUG-08:** The original 16-char (64-bit) truncation has a non-trivial
> collision probability at scale. Use 32 hex chars (128 bits) minimum.

---

### Configuration

| Tool | Purpose |
|------|---------|
| **python-dotenv** | Load `.env` file from project root or `~/.tinirag/.env` |
| **tomllib / tomli** | Read `~/.tinirag/config.toml` (stdlib on 3.11+, tomli on 3.10) |
| **tomli_w** | Write config.toml (tomllib is **read-only** — a separate writer is required) |

> ⚠️ **BUG-04:** Two issues with the original `tomllib` recommendation:
>
> 1. `tomllib` is **read-only** — it can parse TOML but cannot write it.
>    Any `tinirag config set ...` command needs `tomli_w` for writes.
>
> 2. `tomllib` was added in Python **3.11**. The stated minimum is Python 3.10.
>    On 3.10 you must install and import `tomli` (third-party) instead.
>
> **Correct import pattern:**
> ```python
> import sys
> if sys.version_info >= (3, 11):
>     import tomllib
> else:
>     import tomli as tomllib  # pip install tomli
>
> import tomli_w  # pip install tomli_w — used for all config writes
> ```

**Config hierarchy (highest to lowest priority):**
```
CLI flags → environment variables → config.toml → built-in defaults
```

**Example `~/.tinirag/config.toml`:**
```toml
[llm]
endpoint = "http://localhost:11434/v1"
model = "llama3"
temperature = 0
max_tokens = 1024
stream = true

[search]
searxng_url = "http://localhost:8888"
num_results = 5
fetch_top_url = true
fetch_timeout_sec = 2.5   # OPT-02: short per-fetch timeout; fail fast, use snippet fallback

[cache]
enabled = true
ttl_minutes = 10
backend = "sqlite"

[output]
show_sources = true
show_keywords = false
```

---

## Full Dependency List

```
# Core
typer>=0.12
rich>=13.0
httpx>=0.27
openai>=1.30          # use AsyncOpenAI for non-blocking calls

# Search & Extraction
trafilatura>=1.9
markdownify>=0.13

# Token Counting
tiktoken>=0.7         # approximation only for local models — use count_tokens() wrapper

# Config
python-dotenv>=1.0
tomli>=2.0            # BUG-04: required on Python 3.10 (tomllib stdlib only in 3.11+)
tomli_w>=1.0          # BUG-04: tomllib is READ-ONLY; tomli_w required for config writes

# Optional (HuggingFace direct)
huggingface_hub>=0.23
```

**Total install size:** ~45 MB (without torch/transformers — model runtime
is the user's responsibility, not TiniRAG's)

---

## What TiniRAG Does NOT Ship

| Excluded | Reason |
|----------|--------|
| LangChain / LlamaIndex | Too heavy, breaking-change-prone, overkill for CLI |
| Embedding models | Semantic search is out of scope for v1; keyword search sufficient |
| Vector database | No local vector store; stateless per-query retrieval |
| Model runtime | User brings their own (Ollama, vLLM, etc.) |
| Cloud APIs | No OpenAI, Anthropic, or Google API calls — fully local |

---

## Deployment Topology

```
Local Machine
├── tinirag (pip package)        ← user installs this
├── SearXNG (Docker container)   ← user runs this once
└── LLM runtime (user's choice)  ← already running
    ├── Ollama
    ├── vLLM serving HF model
    └── transformers serve
```

**Minimum system requirements:**
- Python 3.10+
- 4 GB RAM (for Phi-3-mini or equivalent small model)
- Docker (for SearXNG) or a remote SearXNG instance URL
- Internet connection (for SearXNG to query external search engines)
