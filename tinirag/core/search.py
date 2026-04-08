"""SearXNG search integration and URL fetching."""

from __future__ import annotations

import asyncio
import re

import httpx
import trafilatura

from tinirag.config import TiniRAGConfig, load_blocklist
from tinirag.core.context import snippet_is_sufficient
from tinirag.core.guardrails import is_stale, is_time_sensitive, log_rail

_HTTP_LIMITS = httpx.Limits(max_connections=10, max_keepalive_connections=5)
_HTTP_HEADERS = {"User-Agent": "TiniRAG/1.0"}


async def check_searxng(url: str) -> bool:
    """GR-R0: Verify SearXNG is reachable before accepting any query."""
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(f"{url}/healthz")
            return r.status_code == 200
    except Exception:
        return False


async def _fetch_url(client: httpx.AsyncClient, url: str, timeout: float) -> str | None:
    """Fetch a single URL and extract clean text (BUG-03: always None-check)."""
    try:
        response = await asyncio.wait_for(
            client.get(url, follow_redirects=True),
            timeout=timeout,
        )
        html = response.text
    except Exception:
        return None

    text = trafilatura.extract(html, include_links=False, include_tables=True)

    # BUG-03: None-check + 100-char minimum (GR-R5)
    if not text or len(text.strip()) < 100:
        return None

    # OPT-08: char-truncate before token counting
    return text[:1600]


async def search_and_fetch(
    keywords: str,
    cfg: TiniRAGConfig,
    raw_query: str = "",
) -> list[dict]:
    """Run SearXNG search + parallel URL fetches. Returns enriched result dicts.

    BUG-01: Validates content-type header before JSON parsing.
    OPT-01: All URL fetches fire simultaneously via asyncio.gather.
    OPT-05: One persistent httpx.AsyncClient per pipeline run.
    """
    blocklist = load_blocklist()
    results: list[dict] = []

    # Faster connect timeout for localhost (managed SearXNG on 127.0.0.1)
    is_local = "127.0.0.1" in cfg.search.searxng_url or "localhost" in cfg.search.searxng_url
    timeout = httpx.Timeout(
        connect=0.5 if is_local else 2.0,
        read=cfg.search.fetch_timeout_sec,
        write=2.0,
        pool=8.0,
    )

    async with httpx.AsyncClient(
        timeout=timeout,
        limits=_HTTP_LIMITS,
        headers=_HTTP_HEADERS,
        follow_redirects=True,
    ) as client:
        # --- SearXNG search ---
        params = {
            "q": keywords,
            "format": "json",
            "categories": "general",
            "pageno": 1,
            "time_range": cfg.search.time_range,  # OPT-07: configurable; "month" faster than "year"
        }
        try:
            resp = await client.get(f"{cfg.search.searxng_url}/search", params=params)
        except Exception as exc:
            raise RuntimeError(f"SearXNG unreachable at {cfg.search.searxng_url}: {exc}") from exc

        # BUG-01: validate content-type BEFORE calling .json()
        content_type = resp.headers.get("content-type", "")
        if "application/json" not in content_type:
            raise RuntimeError(
                "SearXNG returned HTML instead of JSON. "
                "Ensure search.formats includes 'json' in ~/.tinirag/searxng/settings.yml "
                "and run `tinirag stop` to restart.\n"
                f"Content-Type received: {content_type}"
            )

        data = resp.json()
        raw_results = data.get("results", [])[: cfg.search.num_results]

        # Filter blocked domains
        raw_results = [
            r
            for r in raw_results
            if not any(blocked in r.get("url", "").lower() for blocked in blocklist)
        ]

        if not raw_results:
            return []

        # GR-R2: freshness warning — use configured threshold, not hardcoded 180
        freshness_days = cfg.guardrails.source_freshness_days
        stale_keywords = is_time_sensitive(raw_query or keywords)
        if stale_keywords:
            all_stale = all(
                is_stale(r.get("publishedDate"), freshness_days)
                for r in raw_results
                if r.get("publishedDate")
            )
            if all_stale and any(r.get("publishedDate") for r in raw_results):
                log_rail("GR-R2", "stale_sources", keywords)

        # Extract keywords list for snippet sufficiency check
        keyword_list = [kw.strip() for kw in re.split(r"[,\s]+", keywords) if kw.strip()]

        # Decide which URLs to fetch
        urls_to_fetch: list[str | None] = []
        for r in raw_results:
            snippet = r.get("content", "") or r.get("snippet", "")
            if cfg.search.fetch_top_url and not snippet_is_sufficient(snippet, keyword_list):
                urls_to_fetch.append(r.get("url"))
            else:
                urls_to_fetch.append(None)  # snippet is sufficient

        # OPT-01: fire ALL URL fetches simultaneously
        fetch_tasks = [
            _fetch_url(client, url, cfg.search.fetch_timeout_sec)
            if url
            else asyncio.sleep(0, result=None)  # type: ignore[arg-type]
            for url in urls_to_fetch
        ]
        fetched_texts = await asyncio.gather(*fetch_tasks)

        for r, fetched in zip(raw_results, fetched_texts):
            snippet = r.get("content", "") or r.get("snippet", "")
            content = fetched if fetched else snippet
            if not content:
                log_rail("GR-R4", "fetch_failure", keywords, url=r.get("url"))
            results.append({**r, "content": content or snippet})

    return results
