"""Pipeline engine: startup checks, query execution, LLM generation."""

from __future__ import annotations

from dataclasses import dataclass, field

import httpx
from openai import AsyncOpenAI

from tinirag.config import ENDPOINT_PROBE_ORDER, TiniRAGConfig
from tinirag.core.cache import make_cache_key
from tinirag.core.context import build_context
from tinirag.core.guardrails import (
    extract_claims,
    has_injection,
    is_short_response,
    log_rail,
    sensitive_category,
    validate_query,
)
from tinirag.core.optimizer import optimize_query
from tinirag.core.renderer import (
    _collect_stream,
    print_error,
    print_info,
    print_warning,
    stream_response_live,
)
from tinirag.core.search import check_searxng, search_and_fetch
from tinirag.core.session import append_history

# ---------------------------------------------------------------------------
# Module-level functions so tests can patch via patch("tinirag.core.engine.X")
# ---------------------------------------------------------------------------


def is_ollama_running(endpoint: str = "http://localhost:11434") -> bool:
    """Return True if Ollama is reachable at the given endpoint."""
    try:
        r = httpx.get(f"{endpoint}/api/tags", timeout=2.0)
        return r.status_code == 200
    except Exception:
        return False


def check_model_available(model: str, endpoint: str) -> bool:
    """Return True if the model is available at the endpoint."""
    try:
        with httpx.Client(timeout=3.0) as client:  # use context manager to prevent connection leak
            r = client.get(f"{endpoint}/v1/models")
        if r.status_code != 200:
            return False
        data = r.json()
        available = [m["id"] for m in data.get("data", [])]
        # Exact match or prefix match (llama3 → llama3:latest)
        return model in available or any(m.startswith(model + ":") for m in available)
    except Exception:
        return False


def pull_model(model: str, endpoint: str = "http://localhost:11434") -> None:
    """Pull a model via Ollama's pull API (blocks until complete)."""
    try:
        with httpx.Client(timeout=300.0) as client:
            with client.stream("POST", f"{endpoint}/api/pull", json={"name": model}) as r:
                for line in r.iter_lines():
                    if '"status"' in line and '"success"' in line:
                        break
    except Exception as exc:
        raise RuntimeError(f"Failed to pull model '{model}': {exc}") from exc


def probe_endpoints() -> tuple[str, str] | None:
    """Auto-detect a running LLM endpoint. Returns (base_url, runtime_name) or None."""
    for base_url, runtime_name in ENDPOINT_PROBE_ORDER:
        try:
            r = httpx.get(f"{base_url}/models", timeout=1.5)
            if r.status_code == 200:
                return base_url, runtime_name
        except Exception:
            continue
    return None


def _endpoint_base(endpoint: str) -> str:
    """Strip /v1 suffix from an endpoint URL to get the server base URL.

    Uses removesuffix (exact string, not char-set) to avoid mangling ports
    that contain the characters '/', 'v', or '1' (e.g. :11111).
    """
    # Strip trailing slash first so "/v1/" and "/v1" both become the base correctly.
    return endpoint.rstrip("/").removesuffix("/v1")


async def startup_check(cfg: TiniRAGConfig) -> None:
    """Verify Ollama + model are ready. Auto-pull model if missing (GR-R0 for LLM)."""
    endpoint_base = _endpoint_base(cfg.llm.endpoint)

    if not is_ollama_running(endpoint_base):
        # Try to find another running endpoint
        detected = probe_endpoints()
        if detected:
            cfg.llm.endpoint, runtime = detected
            # Recompute endpoint_base from the newly detected endpoint (bug fix: stale base)
            endpoint_base = _endpoint_base(cfg.llm.endpoint)
            print_info(f"Detected {runtime} at {cfg.llm.endpoint}")
        else:
            print_error(
                "No LLM runtime detected. Start Ollama (`ollama serve`) or configure "
                "TINIRAG_ENDPOINT to point to your running endpoint."
            )
            raise SystemExit(1)

    if not check_model_available(cfg.llm.model, endpoint_base):
        print_info(f"Model '{cfg.llm.model}' not found — attempting auto-pull...")
        try:
            pull_model(cfg.llm.model, endpoint_base)
            print_info(f"Model '{cfg.llm.model}' pulled successfully.")
        except RuntimeError as exc:
            print_error(str(exc))
            raise SystemExit(1) from exc


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

_SYSTEM_GROUNDED = (
    "You are a factual assistant. Answer ONLY using the provided context. "
    "If the context does not contain enough information to answer, respond exactly: "
    '"I could not find a reliable answer in the retrieved sources." '  # GR-G1 hardcoded
    "Do not use prior knowledge. Do not speculate. Cite [Source N] for each claim."
)

_SYSTEM_FALLBACK = (
    "You are an honest assistant. You do not have access to real-time web data "
    "for this query. State clearly that you cannot provide a verified answer "
    "and suggest the user search manually."
)


def _build_grounded_messages(context_block: str, query: str) -> list[dict]:
    """BUG-06: context goes in user message, NOT system message."""
    return [
        {"role": "system", "content": _SYSTEM_GROUNDED},
        {
            "role": "user",
            "content": f"Context:\n---\n{context_block}\n---\n\nQuestion: {query}",
        },
    ]


def _build_fallback_messages(query: str) -> list[dict]:
    return [
        {"role": "system", "content": _SYSTEM_FALLBACK},
        {"role": "user", "content": query},
    ]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class QueryResult:
    response: str
    sources: list[dict] = field(default_factory=list)
    keywords: str = ""
    used_search: bool = True
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------


async def run_query(
    raw_query: str,
    cfg: TiniRAGConfig,
    *,
    no_search: bool = False,
    verify: bool = False,
    cache: object = None,
    history: bool = False,
) -> QueryResult:
    """Full RAG pipeline: validate → optimize → search → context → generate.

    Args:
        raw_query: Raw user input string.
        cfg: Loaded TiniRAGConfig.
        no_search: If True, skip retrieval and query LLM directly (GR-G5).
        verify: If True, run hallucination smoke test after generation (GR-G4).
        cache: Cache instance (MemoryCache or SQLiteCache), or None.
        history: If True, append to ~/.tinirag/history.jsonl.
    """
    warnings: list[str] = []

    # --- GR-Q1/Q2: validate query ---
    query = validate_query(raw_query)

    # --- GR-Q3: injection detection (hardcoded, cannot be disabled) ---
    if has_injection(query):
        log_rail("GR-Q3", "injection_detected", query)
        raise ValueError("[!] Query contains instruction injection patterns. Aborted.")

    # --- GR-Q4: sensitive category disclaimer ---
    cat = sensitive_category(query)
    if cat and cfg.guardrails.sensitive_category_disclaimer:
        warnings.append(
            f"This response is for informational purposes only and should not be "
            f"treated as professional {cat} advice."
        )

    llm_client = AsyncOpenAI(
        base_url=cfg.llm.endpoint,
        api_key="not-needed",
    )

    # --- Query optimizer ---
    keywords = await optimize_query(
        query,
        client=llm_client,
        model=cfg.llm.model,
        use_llm=False,  # LLM optimizer disabled by default (CPU-only friendly)
    )

    sources: list[dict] = []
    context_block = ""
    used_search = False

    if not no_search:
        # --- Cache lookup ---
        cache_key = make_cache_key(keywords)
        cached = cache.get(cache_key) if cache else None

        raw_results: list[dict] = []  # always initialise — prevents UnboundLocalError
        if cached is not None:
            raw_results = cached
        else:
            # --- GR-R0: SearXNG health check ---
            if not await check_searxng(cfg.search.searxng_url):
                print_warning(
                    f"SearXNG not reachable at {cfg.search.searxng_url}\n"
                    "    Start it: docker run -d -p 8888:8080 searxng/searxng\n"
                    "    Or set TINIRAG_SEARXNG_URL to your instance."
                )
                no_search = True
            else:
                # --- Search + fetch ---
                try:
                    raw_results = await search_and_fetch(keywords, cfg, raw_query=query)
                except Exception as exc:  # catch all exceptions, not just RuntimeError
                    print_warning(str(exc))
                    raw_results = []

                if cache and raw_results:
                    cache.set(cache_key, raw_results)

        if not no_search:
            # --- GR-R1: zero results handling ---
            if not raw_results:
                # Retry with broader keywords (drop last word)
                words = keywords.split()
                if len(words) > 1:
                    broad_keywords = " ".join(words[:-1])
                    try:
                        raw_results = await search_and_fetch(broad_keywords, cfg, raw_query=query)
                        log_rail("GR-R1", "retry_broad_keywords", keywords)
                    except RuntimeError:
                        raw_results = []

            if raw_results:
                context_block, sources = build_context(
                    raw_results,
                    keywords,
                    cfg.llm.model,
                    max_context_pct=cfg.guardrails.max_context_pct,
                    min_content_chars=cfg.guardrails.min_content_chars,
                    dedup_threshold=cfg.guardrails.dedup_threshold,
                )
                used_search = bool(sources)
            else:
                warnings.append("No web results found. Response based on model knowledge only.")
                log_rail("GR-R1", "zero_results", keywords)

    # --- Build messages ---
    if context_block:
        messages = _build_grounded_messages(context_block, query)
    else:
        messages = _build_fallback_messages(query)

    # --- GR-G5: no-search warning ---
    # Only fire when the user explicitly disabled search or SearXNG was unreachable.
    # When search ran but returned zero useful results, a separate "No web results"
    # warning is already added — "Web search disabled" would be misleading there.
    if no_search:
        warnings.append(
            "Web search disabled. This response uses model training data only. "
            "Knowledge cutoff applies. Verify time-sensitive information independently."
        )

    # --- LLM generation (streaming) ---
    response_text = await _generate(llm_client, messages, cfg)

    # --- GR-G3: short response retry ---
    if is_short_response(response_text):
        log_rail("GR-G3", "short_response", query)
        response_text = await _generate(llm_client, messages, cfg)
        if is_short_response(response_text):
            warnings.append("Model returned an unusually short response. Check model health.")

    # --- GR-G4: hallucination smoke test (slow mode, opt-in) ---
    if verify and sources:
        unverified = extract_claims(response_text, sources)
        for claim in unverified:
            warnings.append(f'Unverified claim: "{claim[:80]}" — not found in sources.')

    # --- History ---
    if history:
        append_history(
            raw_query=query,
            keywords=keywords,
            sources_used=[s.get("url", "") for s in sources],
            response_length=len(response_text),
        )

    return QueryResult(
        response=response_text,
        sources=sources,
        keywords=keywords,
        used_search=used_search,
        warnings=warnings,
    )


async def _generate(
    client: AsyncOpenAI,
    messages: list[dict],
    cfg: TiniRAGConfig,
) -> str:
    """Call the LLM and stream tokens to the terminal. Returns full response."""
    if cfg.llm.stream:
        stream = await client.chat.completions.create(
            model=cfg.llm.model,
            messages=messages,  # type: ignore[arg-type]
            temperature=cfg.llm.temperature,
            max_tokens=cfg.llm.max_tokens,
            stream=True,
        )
        return await stream_response_live(_collect_stream(stream))
    else:
        response = await client.chat.completions.create(
            model=cfg.llm.model,
            messages=messages,  # type: ignore[arg-type]
            temperature=cfg.llm.temperature,
            max_tokens=cfg.llm.max_tokens,
            stream=False,
        )
        text = response.choices[0].message.content or ""
        print(text)
        return text
