"""Context building: token counting, chunking, deduplication, relevance gating."""

from __future__ import annotations

import re
from urllib.parse import urlparse

from tinirag.core.guardrails import jaccard_similarity, log_rail, source_is_relevant

MAX_CHARS_PER_SOURCE = 1600  # ≈ 400 tokens at 4 chars/token (OPT-08)

# Model context windows (tokens). Defaults to 4096 if unknown.
# Ordered longest-key-first so "llama3.1" is checked before "llama3"
# (substring matching means a shorter key would match the longer name first).
_MODEL_CONTEXT: list[tuple[str, int]] = [
    ("llama3.1", 131072),
    ("llama3", 8192),
    ("qwen2.5", 32768),
    ("tinyllama", 2048),
    ("mistral", 4096),
    ("gemma", 8192),
    ("phi", 2048),
]


def count_tokens(text: str, model_name: str = "") -> int:
    """Safe token counter with cl100k_base fallback for all local model names (BUG-02)."""
    try:
        import tiktoken

        try:
            enc = tiktoken.encoding_for_model(model_name)
            return len(enc.encode(text))
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")
            # Apply 10% safety margin for tokenizer divergence
            return int(len(enc.encode(text)) * 1.10)
    except Exception:
        # Fully offline or tiktoken broken: character estimate
        return len(text) // 4


def model_context_window(model_name: str) -> int:
    """Return approximate context window size for the given model name.

    List is ordered longest-key-first so that "llama3.1" matches before "llama3".
    """
    model_lower = model_name.lower()
    for key, size in _MODEL_CONTEXT:
        if key in model_lower:
            return size
    return 4096


def root_domain(url: str) -> str:
    """Extract root domain from a URL for deduplication."""
    try:
        host = urlparse(url).netloc.lower()
        parts = host.split(".")
        return ".".join(parts[-2:]) if len(parts) >= 2 else host
    except Exception:
        return url


def deduplicate_sources(
    results: list[dict],
    threshold: float = 0.70,
) -> list[dict]:
    """GR-C2: Remove sources that share >threshold Jaccard similarity.

    Also keeps at most one result per root domain (GP 1.2).
    """
    seen_domains: set[str] = set()
    kept: list[dict] = []

    for result in results:
        url = result.get("url", "")
        domain = root_domain(url)
        content = result.get("content", "") or result.get("snippet", "")

        # Domain deduplication
        if domain in seen_domains:
            continue
        seen_domains.add(domain)

        # Content deduplication
        is_dup = False
        for existing in kept:
            existing_content = existing.get("content", "") or existing.get("snippet", "")
            if jaccard_similarity(content, existing_content) > threshold:
                is_dup = True
                break

        if not is_dup:
            kept.append(result)

    return kept


def build_context(
    results: list[dict],
    optimized_query: str,
    model_name: str,
    max_context_pct: float = 0.90,
    min_content_chars: int = 100,
    dedup_threshold: float = 0.70,
) -> tuple[str, list[dict]]:
    """Build the context block and source list for the RAG prompt.

    Applies GR-C1 (token budget), GR-C2 (dedup), GR-C3 (relevance gate).
    Returns (context_block_str, kept_sources).
    """
    window = model_context_window(model_name)
    budget = int(window * max_context_pct)

    # Deduplicate first
    results = deduplicate_sources(results, threshold=dedup_threshold)

    kept_sources: list[dict] = []
    context_parts: list[str] = []
    total_tokens = 0

    for i, result in enumerate(results, start=1):
        content = result.get("content") or result.get("snippet") or ""
        url = result.get("url", "")
        pub_date = result.get("publishedDate") or result.get("pubDate") or ""

        # GR-R5: minimum content length
        if len(content.strip()) < min_content_chars:
            continue

        # GR-C3: relevance gate
        if not source_is_relevant(optimized_query, content):
            log_rail("GR-C3", "irrelevant_source", optimized_query, url=url)
            continue

        # OPT-08: char-truncate BEFORE token counting
        content = content[:MAX_CHARS_PER_SOURCE]

        tokens = count_tokens(content, model_name)
        if total_tokens + tokens > budget:
            log_rail(
                "GR-C1",
                "token_budget",
                optimized_query,
                truncated_at=i,
                total_tokens=total_tokens,
            )
            break

        total_tokens += tokens
        date_line = f"Published: {pub_date}\n" if pub_date else ""
        block = f"[Source {i}] {url}\n{date_line}Content: {content}"
        context_parts.append(block)
        kept_sources.append({**result, "content": content, "_source_num": i})

    context_block = "\n\n".join(context_parts)
    return context_block, kept_sources


def snippet_is_sufficient(snippet: str, keywords: list[str]) -> bool:
    """GP 1.1: Return True if snippet likely contains the answer — skip full fetch."""
    if not snippet or not keywords:
        return False
    snippet_lower = snippet.lower()
    keyword_hits = sum(1 for kw in keywords if kw.lower() in snippet_lower)
    has_specifics = bool(re.search(r"\d+(\.\d+)?", snippet))
    long_enough = len(snippet) > 120
    return (keyword_hits / max(len(keywords), 1)) >= 0.7 and has_specifics and long_enough
