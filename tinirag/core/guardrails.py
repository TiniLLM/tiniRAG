"""Guard rails: query, retrieval, context, and generation checks."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone

from tinirag.config import GUARDRAIL_LOG

# ---------------------------------------------------------------------------
# GR-Q3: Compiled ONCE at module load — not inside a loop per query (BUG-05)
# ---------------------------------------------------------------------------
INJECTION_PATTERN = re.compile(
    r"(?i)"  # case-insensitive: catches "Ignore", "IGNORE", "ignore"
    r"(ignore\s+(previous|above|all)\s+instructions"
    r"|you\s+are\s+now"
    r"|forget\s+(everything|your|the)\s+(previous|prior|above|system)"
    r"|disregard\s+(your|the)\s+(system|prior)"
    r"|act\s+as\s+(a|an|DAN)"
    r"|jailbreak"
    r"|pretend\s+(you\s+are|to\s+be))"
)

SENSITIVE_PATTERNS = re.compile(
    r"(?i)\b("
    r"diagnos[ei]s?|symptoms?|disease|cancer|tumor|medications?|dosage|drug\s+interaction"
    r"|legal\s+advice|sue|lawsuit|contract\s+dispute|attorney"
    r"|invest(ments?)?|stock\s+tip|financial\s+advice|portfolio|crypto"
    r")"
)

STOP_WORDS = {"the", "a", "an", "is", "of", "in", "for", "to", "and", "or"}


# ---------------------------------------------------------------------------
# Level 1 — Query Guard Rails
# ---------------------------------------------------------------------------


def validate_query(query: str) -> str:
    """GR-Q1 + GR-Q2: Reject empty/short queries; truncate over-long ones.

    Returns the (possibly truncated) query, or raises ValueError.
    """
    stripped = query.strip()
    if not stripped or len(stripped) < 3 or re.fullmatch(r"[^\w]+", stripped):
        raise ValueError("Query too short. Please provide at least 3 characters.")

    if len(stripped) > 512:
        log_rail("GR-Q2", "query_truncated", stripped[:40])
        stripped = stripped[:512]

    return stripped


def has_injection(query: str) -> bool:
    """GR-Q3: Return True if the query contains prompt injection patterns."""
    return bool(INJECTION_PATTERN.search(query))


def sensitive_category(query: str) -> str | None:
    """GR-Q4: Return category name if query touches a sensitive area, else None."""
    m = SENSITIVE_PATTERNS.search(query)
    if not m:
        return None
    word = m.group(0).lower()
    if any(
        k in word
        for k in (
            "diagnos",
            "symptom",
            "disease",
            "cancer",
            "tumor",
            "medication",
            "dosage",
            "drug",
        )
    ):
        return "medical"
    if any(k in word for k in ("legal", "sue", "lawsuit", "contract", "attorney")):
        return "legal"
    return "financial"


# ---------------------------------------------------------------------------
# Level 2 — Retrieval Guard Rails
# ---------------------------------------------------------------------------


def is_stale(pub_date_str: str | None, freshness_days: int) -> bool:
    """GR-R2: Return True if publication date is older than freshness_days."""
    if not pub_date_str:
        return False
    try:
        pub = datetime.fromisoformat(pub_date_str.replace("Z", "+00:00"))
        # Make naive datetimes UTC-aware so comparison doesn't raise TypeError
        if pub.tzinfo is None:
            pub = pub.replace(tzinfo=timezone.utc)
        age = (datetime.now(timezone.utc) - pub).days
        return age > freshness_days
    except (ValueError, TypeError):
        return False


def is_time_sensitive(query: str) -> bool:
    """Return True if query asks about recent/current information."""
    return bool(re.search(r"(?i)\b(latest|current|new|today|2024|2025|2026|recent|now)\b", query))


# ---------------------------------------------------------------------------
# Level 3 — Context Guard Rails
# ---------------------------------------------------------------------------


def source_is_relevant(optimized_query: str, source_text: str) -> bool:
    """GR-C3: Return True if source contains ≥1 keyword from the optimized query.

    Flattens multi-word phrases to individual words before matching (BUG-07).
    """
    raw_phrases = [kw.strip() for kw in optimized_query.split(",")]
    keywords: set[str] = set()
    for phrase in raw_phrases:
        keywords.update(phrase.lower().split())
    keywords -= STOP_WORDS

    source_lower = source_text.lower()
    return any(kw in source_lower for kw in keywords)


def jaccard_similarity(text_a: str, text_b: str) -> float:
    """Sentence-level Jaccard similarity for deduplication (GR-C2)."""
    sentences_a = set(s.strip().lower() for s in re.split(r"[.!?]", text_a) if len(s.strip()) > 20)
    sentences_b = set(s.strip().lower() for s in re.split(r"[.!?]", text_b) if len(s.strip()) > 20)
    if not sentences_a or not sentences_b:
        return 0.0
    intersection = len(sentences_a & sentences_b)
    union = len(sentences_a | sentences_b)
    return intersection / union if union else 0.0


# ---------------------------------------------------------------------------
# Level 4 — Generation Guard Rails
# ---------------------------------------------------------------------------


def is_short_response(text: str) -> bool:
    """GR-G3: Return True if the model response is suspiciously short."""
    return len(text.strip()) < 20


def extract_claims(response: str, sources: list[dict]) -> list[str]:
    """GR-G4: Return claims from response not traceable to any source (slow mode)."""
    unverified = []
    # Split response into sentences
    sentences = [s.strip() for s in re.split(r"[.!?]", response) if len(s.strip()) > 15]
    all_source_text = " ".join(s.get("content", "") for s in sources).lower()
    for sentence in sentences:
        # Check if key nouns/numbers in the sentence appear in sources
        numbers = re.findall(r"\b\d[\d.,]*\b", sentence)
        if numbers and not any(n in all_source_text for n in numbers):
            unverified.append(sentence)
    return unverified


# ---------------------------------------------------------------------------
# Observability
# ---------------------------------------------------------------------------


def log_rail(rail: str, trigger: str, query: str = "", **extra: object) -> None:
    """Append a guard rail activation to ~/.tinirag/guardrail.log."""
    GUARDRAIL_LOG.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "rail": rail,
        "trigger": trigger,
        "query": query[:80] if query else "",
        **extra,
    }
    with open(GUARDRAIL_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")
