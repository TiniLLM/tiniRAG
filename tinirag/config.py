"""Configuration loading and defaults for TiniRAG."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[no-redef]

import tomli_w

CONFIG_DIR = Path.home() / ".tinirag"
CONFIG_FILE = CONFIG_DIR / "config.toml"
ENV_FILE = CONFIG_DIR / ".env"
BLOCKLIST_FILE = CONFIG_DIR / "blocklist.txt"
GUARDRAIL_LOG = CONFIG_DIR / "guardrail.log"
HISTORY_FILE = CONFIG_DIR / "history.jsonl"
SESSIONS_DIR = CONFIG_DIR / "sessions"

# Default LLM endpoint probe order (port → runtime)
ENDPOINT_PROBE_ORDER = [
    ("http://localhost:11434/v1", "Ollama"),
    ("http://localhost:8000/v1", "vLLM/transformers"),
    ("http://localhost:1234/v1", "LM Studio"),
    ("http://localhost:8080/v1", "llama.cpp"),
]


@dataclass
class LLMConfig:
    endpoint: str = "http://localhost:11434/v1"
    model: str = "llama3"
    temperature: float = 0.0
    max_tokens: int = 1024
    stream: bool = True


@dataclass
class SearchConfig:
    searxng_url: str = "http://localhost:8888"
    num_results: int = 5
    fetch_top_url: bool = True
    fetch_timeout_sec: float = 2.5


@dataclass
class CacheConfig:
    enabled: bool = True
    ttl_minutes: int = 10
    backend: str = "sqlite"  # "sqlite" or "memory"


@dataclass
class OutputConfig:
    show_sources: bool = True
    show_keywords: bool = False


@dataclass
class GuardrailsConfig:
    injection_detection: bool = True
    sensitive_category_disclaimer: bool = True
    source_freshness_days: int = 180
    min_content_chars: int = 100
    max_context_pct: float = 0.90
    dedup_threshold: float = 0.70
    verify_mode: bool = False


@dataclass
class TiniRAGConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    guardrails: GuardrailsConfig = field(default_factory=GuardrailsConfig)


def load_config() -> TiniRAGConfig:
    """Load config from ~/.tinirag/config.toml and environment variables.

    Priority: CLI flags (applied later) > env vars > config.toml > built-in defaults.
    """
    load_dotenv(ENV_FILE)

    cfg = TiniRAGConfig()

    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "rb") as f:
                data = tomllib.load(f)
        except Exception:
            # Malformed TOML — fall back to built-in defaults silently
            data = {}

        if llm := data.get("llm"):
            cfg.llm.endpoint = llm.get("endpoint", cfg.llm.endpoint)
            cfg.llm.model = llm.get("model", cfg.llm.model)
            cfg.llm.temperature = llm.get("temperature", cfg.llm.temperature)
            cfg.llm.max_tokens = llm.get("max_tokens", cfg.llm.max_tokens)
            cfg.llm.stream = llm.get("stream", cfg.llm.stream)

        if search := data.get("search"):
            cfg.search.searxng_url = search.get("searxng_url", cfg.search.searxng_url)
            cfg.search.num_results = search.get("num_results", cfg.search.num_results)
            cfg.search.fetch_top_url = search.get("fetch_top_url", cfg.search.fetch_top_url)
            cfg.search.fetch_timeout_sec = search.get(
                "fetch_timeout_sec", cfg.search.fetch_timeout_sec
            )

        if cache := data.get("cache"):
            cfg.cache.enabled = cache.get("enabled", cfg.cache.enabled)
            cfg.cache.ttl_minutes = cache.get("ttl_minutes", cfg.cache.ttl_minutes)
            cfg.cache.backend = cache.get("backend", cfg.cache.backend)

        if output := data.get("output"):
            cfg.output.show_sources = output.get("show_sources", cfg.output.show_sources)
            cfg.output.show_keywords = output.get("show_keywords", cfg.output.show_keywords)

        if gr := data.get("guardrails"):
            cfg.guardrails.injection_detection = gr.get(
                "injection_detection", cfg.guardrails.injection_detection
            )
            cfg.guardrails.sensitive_category_disclaimer = gr.get(
                "sensitive_category_disclaimer",
                cfg.guardrails.sensitive_category_disclaimer,
            )
            cfg.guardrails.source_freshness_days = gr.get(
                "source_freshness_days", cfg.guardrails.source_freshness_days
            )
            cfg.guardrails.min_content_chars = gr.get(
                "min_content_chars", cfg.guardrails.min_content_chars
            )
            cfg.guardrails.max_context_pct = gr.get(
                "max_context_pct", cfg.guardrails.max_context_pct
            )
            cfg.guardrails.dedup_threshold = gr.get(
                "dedup_threshold", cfg.guardrails.dedup_threshold
            )
            cfg.guardrails.verify_mode = gr.get("verify_mode", cfg.guardrails.verify_mode)

    # Environment variable overrides
    if ep := os.getenv("TINIRAG_ENDPOINT"):
        cfg.llm.endpoint = ep
    if surl := os.getenv("TINIRAG_SEARXNG_URL"):
        cfg.search.searxng_url = surl
    if model := os.getenv("TINIRAG_MODEL"):
        cfg.llm.model = model

    return cfg


def save_config(cfg: TiniRAGConfig) -> None:
    """Write config back to ~/.tinirag/config.toml using tomli_w."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    data = {
        "llm": {
            "endpoint": cfg.llm.endpoint,
            "model": cfg.llm.model,
            "temperature": cfg.llm.temperature,
            "max_tokens": cfg.llm.max_tokens,
            "stream": cfg.llm.stream,
        },
        "search": {
            "searxng_url": cfg.search.searxng_url,
            "num_results": cfg.search.num_results,
            "fetch_top_url": cfg.search.fetch_top_url,
            "fetch_timeout_sec": cfg.search.fetch_timeout_sec,
        },
        "cache": {
            "enabled": cfg.cache.enabled,
            "ttl_minutes": cfg.cache.ttl_minutes,
            "backend": cfg.cache.backend,
        },
        "output": {
            "show_sources": cfg.output.show_sources,
            "show_keywords": cfg.output.show_keywords,
        },
        "guardrails": {
            "injection_detection": cfg.guardrails.injection_detection,
            "sensitive_category_disclaimer": cfg.guardrails.sensitive_category_disclaimer,
            "source_freshness_days": cfg.guardrails.source_freshness_days,
            "min_content_chars": cfg.guardrails.min_content_chars,
            "max_context_pct": cfg.guardrails.max_context_pct,
            "dedup_threshold": cfg.guardrails.dedup_threshold,
            "verify_mode": cfg.guardrails.verify_mode,
        },
    }
    with open(CONFIG_FILE, "wb") as f:
        tomli_w.dump(data, f)


def load_blocklist() -> set[str]:
    """Load user-editable domain blocklist from ~/.tinirag/blocklist.txt."""
    blocked: set[str] = {
        "aicontentfa.com",
    }
    if BLOCKLIST_FILE.exists():
        for line in BLOCKLIST_FILE.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                blocked.add(line.lower())
    return blocked
