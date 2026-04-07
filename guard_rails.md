# TiniRAG — Guard Rails

## Overview

Guard rails are the mechanisms that prevent TiniRAG from producing wrong,
misleading, or low-quality outputs. They operate at four levels:
**query**, **retrieval**, **context**, and **generation**.

Based on current research: RAG alone reduces hallucinations by ~40–71%.
RAG + explicit guard rails reaches 85–96% reduction in well-engineered stacks.

---

## Level 1 — Query Guard Rails

### GR-Q1: Empty Query Rejection
Reject queries that are blank, fewer than 3 characters, or pure punctuation.
```
Error: Query too short. Please provide at least 3 characters.
```

### GR-Q2: Query Length Cap
Truncate queries longer than 512 characters. Log a warning.
Very long queries fragment into poor keywords and slow retrieval.

### GR-Q3: Prompt Injection Detection
Scan the raw user query for known injection patterns before processing.

> ⚠️ **BUG-05 FIX:** The original implementation listed patterns as a plain
> Python list and called `re.search()` in a loop — recompiling all 7 regexes
> on **every single query** with no `re.IGNORECASE`, so `"YOU ARE NOW"` would
> pass through undetected. Use one pre-compiled, case-insensitive pattern:

```python
import re

# Compiled ONCE at module load — not inside a loop per query
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

def has_injection(query: str) -> bool:
    return bool(INJECTION_PATTERN.search(query))
```

If detected: refuse query, print warning, do not pass to LLM.

```
[!] Query contains instruction injection patterns. Aborted.
```

### GR-Q4: Sensitive Category Flag
Flag (but do not block) queries in high-risk categories:
medical diagnosis, legal advice, financial advice.
Append a disclaimer to the response:

```
[!] This response is for informational purposes only and should not be 
treated as professional medical/legal/financial advice.
```

---

## Level 2 — Retrieval Guard Rails

### GR-R0: SearXNG Health Check on Startup (MISSING-01)
Before accepting any query, verify SearXNG is reachable. Without this, a
`ConnectRefused` error mid-pipeline produces an unreadable stack trace
instead of a clear user-facing message.

```python
async def check_searxng(url: str) -> bool:
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(f"{url}/healthz")
            return r.status_code == 200
    except Exception:
        return False

# On CLI startup:
if not await check_searxng(config.searxng_url):
    print(
        "[✗] SearXNG not reachable at " + config.searxng_url + "\n"
        "    Start it: docker run -d -p 8888:8080 searxng/searxng\n"
        "    Or set TINIRAG_SEARXNG_URL to your instance."
    )
    raise SystemExit(1)
```

### GR-R1: Zero Results Handling
If SearXNG returns 0 results:
1. Retry once with broader keywords (drop last keyword)
2. If still 0 results → trigger Layer 3 Fallback Prompt (see prompt.md)
3. Print: `[!] No web results found. Response based on model knowledge only.`
Never silently fall through to LLM-only generation without a warning.

### GR-R2: Source Freshness Check
Parse publication date from SearXNG metadata where available.
If all sources are older than 180 days for a query containing time-sensitive
keywords (latest, current, new, today, 2024, 2025):
```
[!] Retrieved sources may be outdated. Verify independently for latest info.
```

### GR-R3: Domain Blocklist
Never include content from:
```python
BLOCKED_DOMAINS = [
    # Known hallucination-heavy AI content farms
    "aicontentfa.com",
    # Paywalled content that returns 403 (misleading empty context)
    # Domains in user-configurable blocklist ~/.tinirag/blocklist.txt
]
```

Load a user-editable `~/.tinirag/blocklist.txt` for custom domain exclusions.

### GR-R4: Fetch Failure Graceful Degradation
If a URL fetch fails (timeout, 4xx, 5xx):
- Fall back to SearXNG snippet for that source
- Do not crash the pipeline
- Log: `[warn] Failed to fetch {url}: {reason}. Using snippet only.`

### GR-R5: Content Length Sanity Check
If extracted text from a URL is < 100 characters after cleaning:
discard that source (likely a paywall, redirect, or error page).
Minimum viable content threshold: 100 characters.

---

## Level 3 — Context Guard Rails

### GR-C1: Token Budget Enforcement
Before sending the final prompt to the LLM, count tokens.
If total tokens > 90% of model context window:
1. Remove the lowest-ranked source first
2. Truncate remaining sources proportionally
3. Never truncate the system prompt or user query
4. Log: `[warn] Context truncated to fit model window ({n} tokens).`

**Token counting formula:**
```python
total = len(system_prompt_tokens) + len(context_tokens) + len(query_tokens)
budget = model_context_window * 0.90
```

### GR-C2: Duplicate Content Deduplication
Before injecting context, compute character-level similarity between sources.
If two sources share > 70% overlapping content (Jaccard on sentences):
keep only the higher-ranked source, discard the duplicate.

### GR-C3: Context Relevance Gate
Before injecting a source into the prompt, verify it contains at least one
keyword from the optimized query. If a source shares zero keywords with the
query, discard it — it was retrieved by false association.

> ⚠️ **BUG-07 FIX:** The original split on `","` produced multi-word phrases
> like `"macbook neo"` as single entries. Substring matching a two-word phrase
> fails when the words appear in different sentences in the source text, causing
> valid sources to be incorrectly discarded. Flatten to individual words first:

```python
STOP_WORDS = {"the", "a", "an", "is", "of", "in", "for", "to", "and", "or"}

def source_is_relevant(optimized_query: str, source_text: str) -> bool:
    """
    Returns True if source contains at least one keyword from the query.
    Flattens multi-word phrases to individual words before matching.
    BUG-07: "macbook neo" split(",") gave a phrase; now gives {"macbook","neo"}.
    """
    # Flatten "macbook neo, GPU architecture" → {"macbook","neo","gpu","architecture"}
    raw_phrases = [kw.strip() for kw in optimized_query.split(",")]
    keywords = set()
    for phrase in raw_phrases:
        keywords.update(phrase.lower().split())
    keywords -= STOP_WORDS  # remove noise words

    source_lower = source_text.lower()
    return any(kw in source_lower for kw in keywords)

# Usage
if not source_is_relevant(optimized_query, source_text):
    skip_source()
```

---

## Level 4 — Generation Guard Rails

### GR-G1: Mandatory Abstention Instruction
The system prompt MUST contain the abstention clause:

```
If the context does not contain enough information to answer, respond exactly:
"I could not find a reliable answer in the retrieved sources."
```

This is the single highest-impact guard rail. It prevents the model from
confabulating when context is insufficient.

### GR-G2: Source Citation Enforcement
Instruct the model to cite sources:
```
Cite source URLs as [Source N] when referencing specific claims.
```
Post-process the response to verify at least one `[Source N]` citation
appears for factual claims. If no citation found in a factual response,
append the sources block manually.

### GR-G3: Response Length Sanity
If the model returns a response shorter than 20 characters:
treat it as a generation failure and retry once.
If the second response is also < 20 characters:
print `[!] Model returned an unusually short response. Check model health.`

### GR-G4: Hallucination Smoke Test (Optional, Slow Mode)
When `--verify` flag is enabled:
1. Extract key claims from the model response (nouns + numbers)
2. Check each claim appears in at least one source
3. Flag claims not traceable to any source:
```
[?] Unverified claim: "A18 Pro chip uses 6-core GPU" — not found in sources.
```
This adds ~1–2 seconds per response. Disabled by default.

### GR-G5: Recency Warning on Stale Model Knowledge
If `--no-search` mode is active:
always prepend the response with:
```
[!] Web search disabled. This response uses model training data only.
    Knowledge cutoff applies. Verify time-sensitive information independently.
```

---

## Error State Hierarchy

```
Query received
    │
    ├── GR-Q1/Q2 fails  → Hard reject with user message
    ├── GR-Q3 fails     → Hard reject (injection detected)
    │
    ├── GR-R1 fails     → Retry, then fallback to LLM-only + warning
    ├── GR-R4 fails     → Use snippet + warning (never crash)
    │
    ├── GR-C1 fails     → Truncate context + warning (never crash)
    │
    ├── GR-G3 fails     → Retry once, then surface model health warning
    │
    └── All guards pass → Normal response + sources block
```

---

## Guard Rail Configuration

All guard rails are configurable in `~/.tinirag/config.toml`:

```toml
[guardrails]
injection_detection = true         # GR-Q3
sensitive_category_disclaimer = true  # GR-Q4
source_freshness_days = 180        # GR-R2
min_content_chars = 100            # GR-R5
max_context_pct = 0.90             # GR-C1
dedup_threshold = 0.70             # GR-C2
verify_mode = false                # GR-G4 (slow mode)
```

Power users can disable individual guard rails, but cannot disable GR-Q3
(injection detection) or GR-G1 (abstention instruction) — these are hardcoded.

---

## Guard Rail Non-Negotiables

The following two guard rails are **hardcoded and cannot be disabled**:

1. **GR-Q3 — Prompt Injection Detection**
   TiniRAG is a pipeline tool. Injection attacks in the query could
   propagate to the LLM and cause policy violations or data leakage.

2. **GR-G1 — Abstention Instruction**
   Without this, TiniRAG is no better than a vanilla LLM call.
   The entire value proposition of RAG depends on the model knowing
   when to say "I don't know."

---

## Observability for Guard Rails

Every guard rail activation is logged to `~/.tinirag/guardrail.log`:

```jsonl
{"ts":"2025-04-07T10:23:11","rail":"GR-R2","trigger":"stale_sources","query":"macbook neo gpu"}
{"ts":"2025-04-07T10:23:45","rail":"GR-C1","trigger":"token_budget","truncated_sources":2}
```

This log enables debugging when responses feel wrong.
Run `tinirag logs --rails` to view recent activations.
