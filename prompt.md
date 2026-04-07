# TiniRAG — Prompt Design Guide

## Overview

This document defines the prompt architecture for TiniRAG's CLI pipeline.
The system uses **three prompt layers**: Query Optimizer, Retrieval Grounding,
and Final Generation. All prompts are model-agnostic and work over any
OpenAI-compatible `/v1/chat/completions` endpoint.

---

## Layer 1 — Query Optimizer Prompt

**Purpose:** Distill a natural language user query into short, high-signal
search keywords before sending to SearXNG. Reduces retrieval noise and speeds
up search.

**When to trigger:** Tiered — regex first, LLM optimizer only as fallback.

**Tier 1 — Regex optimizer (instant, no model call, handles ~80% of queries):**
```python
import re

STRIP_PATTERNS = re.compile(
    r"(?i)\b(what|how|why|when|where|who|which|tell me|explain|"
    r"describe|can you|please|i want to know|i need|find out|"
    r"give me|show me|is there|are there|does|do|will|would|"
    r"about|regarding)\b"
)

def regex_optimize(query: str) -> str:
    cleaned = STRIP_PATTERNS.sub("", query)
    cleaned = re.sub(r"\s+", " ", cleaned).strip().rstrip("?.!,")
    return cleaned if len(cleaned) >= 3 else query
```

**Tier 2 — LLM optimizer (only when regex result is still > 8 words):**

**Recommended model:** A small/fast model (e.g. Qwen2.5-0.5B, Phi-3-mini).
Only invoke if `len(regex_result.split()) > 8`. Skip entirely on CPU-only machines.

```
SYSTEM:
You are a search query optimizer. Extract the most important nouns, named
entities, and technical terms from the user query. Return ONLY 3–6 keywords
as a comma-separated list. No explanation. No punctuation beyond commas.

USER:
{regex_optimized_query}
```

**Input:** Raw user query string → regex strip → optional LLM pass
**Output:** `macbook neo, GPU architecture, Apple silicon`

**Rules:**
- Always run Tier 1 first — saves 500ms–2s per query
- Call Tier 2 only when Tier 1 output > 8 words
- Strip question words (what, how, why, when)
- Strip filler words (tell me about, explain, I want to know)
- Keep product names, version numbers, technical nouns intact
- If query is already short (<= 5 words), pass it through unchanged

---

## Layer 2 — Retrieval Grounding Prompt (Core RAG Prompt)

**Purpose:** Feed retrieved web context to the LLM and constrain generation
to only what was retrieved. This is the primary anti-hallucination layer.

**Template:**
```
SYSTEM:
You are a factual assistant. Answer ONLY using the provided context.
If the context does not contain enough information to answer, respond exactly:
"I could not find a reliable answer in the retrieved sources."
Do not use prior knowledge. Do not speculate. Cite [Source N] for each claim.

USER:
Context:
---
{retrieved_context}
---

Question: {original_user_query}
```

> ⚠️ **BUG-06 FIX:** Context is placed in the **user** message, NOT the system
> message. Injecting long context into the system message breaks Mistral-family
> models and older llama.cpp builds that do not correctly propagate long system
> messages. All major open-source RAG projects (Open WebUI, Dify, AnythingLLM)
> use this user-message pattern for cross-model reliability.

**Context block format:**
```
[Source 1] URL: https://example.com/article
Content: <snippet or extracted text, max 400 tokens>

[Source 2] URL: https://anothersite.com/page
Content: <snippet or extracted text, max 400 tokens>
```

**Key constraints:**
- `max_tokens` for context block: 1500–2000 total (across all sources)
- Number of sources: 3–5 recommended, 7 maximum
- If a source is fetched (full page), truncate to 400 tokens per source
- If only snippets available, include up to 5 snippets

---

## Layer 3 — Fallback / No-Context Prompt

**Purpose:** Used when SearXNG returns zero results or all fetches fail.
Prevents silent hallucination.

```
SYSTEM:
You are an honest assistant. You do not have access to real-time web data
for this query. State clearly that you cannot provide a verified answer
and suggest the user search manually.

USER:
{original_user_query}
```

---

## Prompt Chaining Flow

```
User Input
    │
    ▼
[Tier 1] Regex Optimizer  ──► fast keywords (<1ms)
    │
    ├── result > 8 words?
    │       └── YES ──► [Tier 2] LLM Optimizer (~500ms)
    │       └── NO  ──► use Tier 1 result directly
    │
    ▼
SearXNG Search (keywords) + asyncio.gather(URL fetches)  ← parallel
    │
    ├── results found ──► [Layer 2] Grounded Generation (stream=True, flush=True)
    │
    └── no results    ──► [Layer 3] Fallback Response
```

---

## Model-Agnostic Compatibility Notes

TiniRAG targets any model served via an OpenAI-compatible endpoint:

| Runtime          | Endpoint                          | Notes                    |
|------------------|-----------------------------------|--------------------------|
| Ollama           | `http://localhost:11434/v1`       | Most common local setup  |
| vLLM             | `http://localhost:8000/v1`        | HuggingFace models       |
| transformers serve | `http://localhost:8000/v1`      | HF native server         |
| LM Studio        | `http://localhost:1234/v1`        | GUI-based local runner   |
| llama.cpp server | `http://localhost:8080/v1`        | Lightweight GGUF runtime |
| HF TGI           | `http://<host>:80/v1`             | Production HF endpoint   |

All use the same `POST /v1/chat/completions` interface with identical prompt
structure. Configure via `--endpoint` flag or `TINIRAG_ENDPOINT` env variable.

---

## Prompt Engineering Best Practices Applied

1. **System role sets hard constraints** — "Answer ONLY using context" is the
   most effective single instruction for reducing hallucination in RAG.

2. **Abstention clause is mandatory** — Models must be told explicitly when
   to say "I don't know." Without this, all models will fill gaps with
   confident fabrications.

3. **Context goes in the user message, not the system message** — Injecting
   the retrieved context block into the system message breaks Mistral-family
   models and older llama.cpp builds. Always put context in the user turn
   with clear `---` delimiters around it. (BUG-06)

4. **Tiered query optimizer** — Always run the regex stripper first (< 1ms).
   Only call a small LLM optimizer when the regex output is still > 8 words.
   This saves 500ms–2s on ~80% of queries. (OPT-03)

5. **Short optimizer prompts** — The query optimizer prompt is intentionally
   minimal. Longer instructions degrade keyword extraction quality.

6. **No few-shot in grounding prompt** — Adding examples increases token cost
   with minimal accuracy gain for standard factual queries.

7. **Separate system and user roles** — Never merge system instructions and
   user content into a single message role.
