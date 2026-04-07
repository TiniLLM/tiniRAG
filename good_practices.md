# TiniRAG — Good Practices

## Overview

Accumulated best practices from the open-source community (Reddit r/LocalLLaMA,
GitHub discussions, Anthropic Engineering Blog, Hugging Face blog, DEV.to) for
building reliable, fast, model-agnostic RAG CLI pipelines.

---

## 1. Retrieval Practices

### 1.1 Always use snippet-first, fetch-on-demand
SearXNG returns snippets (~150–200 chars) by default. These are fast but often
too shallow for technical queries.

**Pattern:**
```
snippets only   → use for factual lookups (who, what, when)
fetch top URL   → use for technical/architecture queries
fetch top 2–3   → use for comparative or research queries
```

Run snippets first. Use this concrete sufficiency check before firing any
URL fetch — saves 1–3s on ~40% of queries:

```python
import re

def snippet_is_sufficient(snippet: str, keywords: list[str]) -> bool:
    """
    Returns True if the snippet likely already contains the answer.
    Avoids a full URL fetch when not needed.
    """
    snippet_lower = snippet.lower()
    keyword_hits = sum(1 for kw in keywords if kw in snippet_lower)
    has_specifics = bool(re.search(r"\d+(\.\d+)?", snippet))  # has version/number
    long_enough = len(snippet) > 120

    # ≥70% keyword hit + specific data + sufficient length → skip fetch
    return (keyword_hits / max(len(keywords), 1)) >= 0.7 and has_specifics and long_enough
```

For factual queries ("who founded X", "what version is Z"), snippets almost
always contain the answer. Fetch only when `snippet_is_sufficient()` returns False.

### 1.2 Deduplicate retrieved content
Multiple SearXNG sources often return overlapping text (syndicated articles).
Always deduplicate by URL domain before building the context block.
Keep at most one result per root domain.

### 1.3 Prefer primary sources over aggregators
Rank sources in this order:
1. Official docs / release notes (docs.*, developer.*)
2. GitHub repositories and releases
3. Research papers (arxiv.org, papers.*)
4. Tech blogs (Anthropic, HuggingFace, Google DeepMind blogs)
5. News coverage (The Verge, Ars Technica, TechCrunch)
6. Community discussions (Reddit, Hacker News) — last resort only

Deprioritise: Medium SEO farms, listicle aggregators, copied-content sites.

### 1.4 Set result count carefully
- Default: `num_results=5` in SearXNG
- For simple fact queries: `num_results=3` is sufficient
- For broad research: `num_results=7`, then re-rank by domain quality
- Never exceed 10 — marginal results add noise, not signal

---

## 2. Context Construction Practices

### 2.1 Respect token budgets
Total context window occupation should not exceed 60% of the model's limit.
Reserve 40% for the model's reasoning and response generation.

| Model context | Max context block | Notes                          |
|---------------|-------------------|--------------------------------|
| 2048 tokens   | 1000 tokens       | Small models (Phi-2, TinyLlama)|
| 4096 tokens   | 2000 tokens       | Standard (Mistral 7B, Llama 3) |
| 8192 tokens   | 4500 tokens       | Extended (Llama 3.1, Qwen2.5)  |
| 32k+          | 8000 tokens       | Large context models           |

### 2.2 Chunk fetched pages meaningfully
When fetching a full URL, do not blindly truncate at N characters.
Extract the meaningful section:
- Strip nav, footer, cookie banners, ads
- Locate the first H1/H2 that matches a keyword
- Extract 3–5 paragraphs below it
- Truncate at sentence boundaries, not mid-word

Use `trafilatura` or `BeautifulSoup` for HTML extraction.
Convert to plain text or markdown before injecting into prompt.

**Always character-truncate BEFORE token counting (OPT-08):**
```python
# Cheap char truncation first (~instant)
MAX_CHARS_PER_SOURCE = 1600  # ≈ 400 tokens at 4 chars/token
source_text = source_text[:MAX_CHARS_PER_SOURCE]

# Now token-count the already-small text (fast)
token_count = count_tokens(source_text, model_name)
```
Counting tokens on a full fetched page (5000+ chars) then discarding 80% of
it wastes ~60% of token-counting overhead. Truncate by character first.

### 2.3 Label sources clearly in context
```
[Source 1] https://example.com
Published: 2025-03-15
Content: ...

[Source 2] https://anothersite.com
Published: 2024-11-02
Content: ...
```
The model will use labels when generating citations. Recency metadata helps
models weigh newer sources correctly.

---

## 3. Query Optimization Practices

### 3.1 Keyword distillation is the highest-ROI optimization
Converting "What GPU does the MacBook Neo use?" to `macbook neo GPU chip`
reduces SearXNG latency by 30–50% and improves result relevance.

**Use a tiered optimizer — regex first, LLM only as fallback:**

```python
import re

STRIP_PATTERNS = re.compile(
    r"(?i)\b(what|how|why|when|where|who|which|tell me|explain|"
    r"describe|can you|please|i want to know|i need|find out|"
    r"give me|show me|is there|are there|does|do|will|would|"
    r"about|regarding)\b"
)

def optimize_query(query: str, use_llm: bool = False) -> str:
    # Tier 1: regex strip (always, <1ms)
    cleaned = STRIP_PATTERNS.sub("", query)
    cleaned = re.sub(r"\s+", " ", cleaned).strip().rstrip("?.!,")
    result = cleaned if len(cleaned) >= 3 else query

    # Tier 2: LLM pass (only when still verbose after regex)
    if use_llm and len(result.split()) > 8:
        result = llm_optimize(result)  # calls small model
    return result
```

The regex stripper alone handles ~80% of queries in < 1ms.
Only call a small LLM optimizer for queries still > 8 words after stripping.
This saves 500ms–2s per query on the common path.

### 3.2 Detect entity type before search
Classify the query intent to pick the right SearXNG category:

| Intent type        | SearXNG category  | Example                        |
|--------------------|-------------------|--------------------------------|
| Product/hardware   | `general`         | MacBook Neo GPU specs          |
| News/current event | `news`            | Apple event announcements      |
| Code/technical     | `general`         | vLLM OpenAI endpoint setup     |
| Scientific         | `science`         | transformer architecture paper |

### 3.3 Add temporal context to time-sensitive queries
If the query contains words like "latest", "new", "current", "2025", append
the current year to keywords:
`macbook neo GPU architecture 2025`

---

## 4. LLM Integration Practices

### 4.1 Use the OpenAI-compatible `/v1/chat/completions` endpoint universally
All major local runtimes support this interface:
Ollama, vLLM, llama.cpp, LM Studio, transformers serve, TGI.
Never use runtime-specific APIs — they break portability.

### 4.2 Detect the running endpoint automatically
On startup, TiniRAG should probe these ports in order:
```
11434  → Ollama
8000   → vLLM / transformers serve
1234   → LM Studio
8080   → llama.cpp server
```
First successful `GET /v1/models` wins. Surface detected runtime and model
name to the user before querying.

### 4.3 Always set `temperature=0` for factual RAG
Factual retrieval-grounded queries need deterministic outputs.
Never use temperature > 0.3 in a RAG pipeline.
Reserve higher temperature for creative or summarisation tasks.

### 4.4 Stream output correctly — flush=True is mandatory
For CLI UX, always stream (`stream=True`) so the user sees tokens as they
arrive. This reduces perceived latency significantly on slower local models.

**Critical implementation detail — `flush=True`:**
```python
async def stream_response(client, messages: list, model: str) -> str:
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        stream=True,
    )
    full_response = []
    async for chunk in response:
        delta = chunk.choices[0].delta.content
        if delta:
            print(delta, end="", flush=True)  # ← flush=True is CRITICAL
            full_response.append(delta)
    print()  # final newline
    return "".join(full_response)
```

Without `flush=True`, Python buffers stdout until the buffer fills (~4 KB).
The user sees nothing until the entire response is done — streaming appears
broken. This is the **most commonly reported "streaming doesn't work" bug**
on r/LocalLLaMA for CLI LLM tools.

### 4.5 Handle model context overflow gracefully
Before sending, count approximate tokens (1 token ≈ 4 characters).
If context + query > 90% of model limit, truncate oldest/lowest-ranked
sources first, never truncate the user query.

---

## 5. Pipeline-Level Practices

### 5.1 Cache search results with TTL — normalize keys properly
Many queries repeat within a session. Cache SearXNG results keyed by
normalized keywords with a 10-minute TTL (in-memory via dict or SQLite).
This eliminates redundant network calls for the same query rephrased.

**Key normalization (OPT-06) — without this, `"GPU MacBook"` and `"macbook gpu"` are separate cache entries:**
```python
import re, hashlib

def normalize_for_cache(keywords: str) -> str:
    s = keywords.lower()
    s = re.sub(r"[^\w\s]", "", s)           # strip punctuation
    words = sorted(s.split())               # sort so order doesn't matter
    NOISE = {"a", "an", "the", "is", "of"}
    words = [w for w in words if w not in NOISE]
    return " ".join(words)

# Use 32 hex chars (128-bit) — not 16 (BUG-08: 64-bit has collision risk)
cache_key = hashlib.sha256(normalize_for_cache(keywords).encode()).hexdigest()[:32]
```

### 5.2 Log queries and responses (opt-in)
Maintain a local `~/.tinirag/history.jsonl` file per session.
Format: `{timestamp, raw_query, keywords, sources_used, response_length}`.
Never log full response text unless user opts in.
This data enables future fine-tuning and quality analysis.

### 5.3 Show provenance in every response
Always append a sources block at the end of every response:
```
─── Sources ───────────────────────────────────────
[1] https://example.com/article (retrieved 2025-04-07)
[2] https://anothersite.com/page (retrieved 2025-04-07)
───────────────────────────────────────────────────
```
This is non-negotiable. It differentiates TiniRAG from plain LLM output and
builds user trust in the answers.

### 5.4 Expose a `--no-search` flag
Let users bypass retrieval and query the LLM directly.
Useful for: creative tasks, code generation, non-factual queries.
Always print a warning: `[!] Web search disabled. Response may be outdated.`

---

## 6. Community-Validated Observations

From r/LocalLLaMA, GitHub discussions, and Hacker News threads:

- **SearXNG JSON API requires explicit opt-in.** The Docker image ships with
  JSON format **disabled** in `settings.yml`. Without `search: formats: [json]`
  in settings.yml, `?format=json` silently returns HTML. Always validate
  `content-type: application/json` in the response before parsing. (BUG-01)

- **Snippet quality varies by engine.** Google snippets > Bing > DuckDuckGo
  for technical content. Configure SearXNG to prioritise Google + Brave.

- **Parallel URL fetching is the #1 speed win.** Using `asyncio.gather` to
  fire all URL fetches simultaneously reduces 5× 1s fetches from 5s sequential
  to ~1s total. This is the highest-impact optimisation in the entire pipeline.

- **`flush=True` is required for CLI streaming.** Without it, stdout buffers
  until ~4KB fills and the user sees no output until the response is complete.
  This is the most commonly reported "streaming broken" issue for local LLM CLIs.

- **Small models hallucinate more when context is long.** Keep context blocks
  under 1000 tokens for models ≤ 7B parameters.

- **The abstention instruction is critical.** Without "say I don't know",
  models will fabricate answers from retrieved unrelated context.

- **Re-ranking matters more than retrieval count.** Top-3 well-ranked results
  beat top-10 unranked results for response quality.
