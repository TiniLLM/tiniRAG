"""Query optimizer: tiered regex + optional LLM pass."""

from __future__ import annotations

import re

from openai import AsyncOpenAI

# Tier 1: compiled ONCE at module load
STRIP_PATTERNS = re.compile(
    r"(?i)\b(what|how|why|when|where|who|which|tell\s+me|explain|"
    r"describe|can\s+you|please|i\s+want\s+to\s+know|i\s+need|find\s+out|"
    r"give\s+me|show\s+me|is\s+there|are\s+there|does|do|will|would|"
    r"about|regarding)\b"
)


def regex_optimize(query: str) -> str:
    """Tier 1 optimizer: strip question/filler words in <1ms."""
    cleaned = STRIP_PATTERNS.sub("", query)
    cleaned = re.sub(r"\s+", " ", cleaned).strip().rstrip("?.!,")
    return cleaned if len(cleaned) >= 3 else query


async def llm_optimize(keywords: str, client: AsyncOpenAI, model: str) -> str:
    """Tier 2 optimizer: call a small LLM to distill to 3–6 keywords.

    Only invoked when regex result is still > 8 words.
    """
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a search query optimizer. Extract the most important nouns, "
                        "named entities, and technical terms from the user query. Return ONLY "
                        "3–6 keywords as a comma-separated list. No explanation. No punctuation "
                        "beyond commas."
                    ),
                },
                {"role": "user", "content": keywords},
            ],
            temperature=0,
            max_tokens=64,
            stream=False,
        )
        result = response.choices[0].message.content or keywords
        return result.strip()
    except Exception:
        # Fallback to regex result if LLM optimizer fails
        return keywords


async def optimize_query(
    query: str,
    *,
    client: AsyncOpenAI | None = None,
    model: str | None = None,
    use_llm: bool = False,
) -> str:
    """Full tiered optimizer: regex first, LLM only if still verbose."""
    result = regex_optimize(query)

    if use_llm and client is not None and model is not None and len(result.split()) > 8:
        result = await llm_optimize(result, client, model)

    return result
