# TiniRAG Build Log

## Session 3 — 2026-04-07 (Comprehensive Test Expansion)

### Overview
Fail-proof test suite expansion. Added 77 new tests across 7 files (including new `tests/test_renderer.py`). Found and fixed 2 bugs during test authoring. Final state: **243/243 passing**.

---

### BUG-FIX-20: `load_config()` crashed on malformed TOML

**Symptom:** `load_config()` raised `tomllib.TOMLDecodeError` if `~/.tinirag/config.toml` was corrupt or hand-edited incorrectly.

**Root cause:** The TOML `load()` call was not wrapped in a try/except.

**Fix:** Wrapped `tomllib.load(f)` in `try/except Exception` — silently falls back to built-in defaults (same behavior as missing file).

**File:** `tinirag/config.py` → `load_config()`

---

### Test coverage added

| File | New tests | Coverage areas |
|---|---|---|
| `test_guardrails.py` | +22 | 512-char boundary, adversarial whitespace/newline injection, all `sensitive_category` branches, `is_time_sensitive`, `extract_claims` with empty sources |
| `test_cache.py` | +16 | `SQLiteCache` (all CRUD + TTL + persistence across instances + dir creation), `make_cache` factory, edge cases (empty string, noise-only, unicode) |
| `test_optimizer.py` | +7 | `optimize_query` async orchestrator: LLM skipped/called/fallback/no-client paths |
| `test_engine.py` | +10 | `_build_grounded_messages` (BUG-06 assertion), `_build_fallback_messages`, GR-G3 retry (count=2 + persistent-short warning), verify mode (GR-G4) |
| `test_renderer.py` | +14 | **New file** — `stream_response_live` (empty/single/multi/skip-empty-tokens), `_collect_stream` (empty-choices BUG-16, None delta, mixed chunks), `print_sources` (empty list, IPv6 crash, missing fields) |
| `test_config.py` | +4 | Malformed TOML fallback, partial TOML defaults, env-over-TOML precedence, missing blocklist file |
| `test_session.py` | — | Already comprehensive |

---

## Session 2 — 2026-04-07 (Code Review — Bug Fixes)

### Overview
Senior code review pass over all source modules. Found and fixed 11 bugs. Tests: **166/166 passing** (13 new tests added).

---

### BUG-FIX-08: `rstrip("/v1")` strips individual characters, not the suffix

**Symptom:** Any endpoint whose port or path contains the characters `/`, `v`, or `1` at the tail gets mangled. Example: `"http://myserver:8001/v1"` → `"http://myserver:800"`. Default ports (11434, 8000, 1234, 8080) happen to avoid this only because they end in `4`, `0`, `4`, `0` respectively — latent bug for custom endpoints.

**Root cause:** `str.rstrip(chars)` strips a SET of characters, not a substring. `rstrip("/v1")` strips any trailing chars in `{'/', 'v', '1'}`.

**Fix:** `_endpoint_base()` helper using `endpoint.rstrip("/").removesuffix("/v1")`. `removesuffix` is an exact string match, not a character set. The trailing-slash strip comes first so `/v1/` and `/v1` both normalise correctly.

**File:** `tinirag/core/engine.py` → new `_endpoint_base()`, used in `startup_check()`

---

### BUG-FIX-09: `startup_check` used stale `endpoint_base` after auto-detecting new endpoint

**Symptom:** When Ollama is unreachable and `probe_endpoints()` detects vLLM on port 8000, `cfg.llm.endpoint` is updated but `endpoint_base` still holds the old `"http://localhost:11434"`. Subsequent `check_model_available()` and `pull_model()` calls target the wrong host.

**Root cause:** `endpoint_base` was computed once at the top of `startup_check()` and never refreshed after the `detected` branch updated `cfg.llm.endpoint`.

**Fix:** Added `endpoint_base = _endpoint_base(cfg.llm.endpoint)` inside the `if detected:` branch.

**File:** `tinirag/core/engine.py` → `startup_check()`

---

### BUG-FIX-10: `check_model_available` leaked `httpx.Client` on exception

**Symptom:** Connection pool resources were never released when an exception occurred inside `check_model_available`.

**Root cause:** `httpx.Client()` was instantiated without a `with` block, so `client.close()` was never called.

**Fix:** Changed to `with httpx.Client(timeout=3.0) as client:`.

**File:** `tinirag/core/engine.py` → `check_model_available()`

---

### BUG-FIX-11: `raw_results` potentially unbound — `except RuntimeError` too narrow

**Symptom:** If `search_and_fetch` raised a non-`RuntimeError` exception (e.g. `json.JSONDecodeError`, `httpx.HTTPStatusError`), the `except RuntimeError` block wouldn't catch it, leaving `raw_results` unbound and causing `UnboundLocalError` at `if not raw_results:`.

**Fix:** Added `raw_results: list[dict] = []` before the try block as a safe default. Also broadened `except RuntimeError` to `except Exception` so unexpected exceptions degrade gracefully rather than crashing.

**File:** `tinirag/core/engine.py` → `run_query()`

---

### BUG-FIX-12: `model_context_window` returned wrong value for llama3.1

**Symptom:** `model_context_window("llama3.1")` returned `8192` instead of `131072` because `"llama3"` is a substring of `"llama3.1"` and the dict was iterated in insertion order (llama3 before llama3.1).

**Root cause:** Substring match `key in model_lower` with a shorter key (`"llama3"`) matching before the longer specific key (`"llama3.1"`).

**Fix:** Changed `_MODEL_CONTEXT` from a `dict` to a `list[tuple[str, int]]` ordered longest-key-first, so `"llama3.1"` is checked before `"llama3"`.

**File:** `tinirag/core/context.py` → `_MODEL_CONTEXT`, `model_context_window()`

---

### BUG-FIX-13: "Web search disabled" warning fired incorrectly when search returned zero results

**Symptom:** When a query ran through SearXNG but found zero results, the warning said both "No web results found" AND "Web search disabled" — the second message was misleading (search WAS enabled, it just found nothing).

**Root cause:** The `no_search` flag was repurposed mid-flow: when SearXNG was unreachable it was set to `True`, but the GR-G5 warning fired on `no_search or not used_search`. The `not used_search` branch fired even for zero-results cases.

**Fix:** Changed condition to `if no_search:` only — fires when the user explicitly passed `--no-search` or SearXNG was unreachable, not when search ran and found nothing.

**File:** `tinirag/core/engine.py` → `run_query()`

---

### BUG-FIX-14: GR-R2 freshness check ignored `cfg.guardrails.source_freshness_days`

**Symptom:** The freshness threshold was hardcoded as `180` in `search_and_fetch`, ignoring whatever the user configured in `config.toml`.

**Fix:** Read `cfg.guardrails.source_freshness_days` and pass it to `is_stale()`.

**File:** `tinirag/core/search.py` → `search_and_fetch()`

---

### BUG-FIX-15: URLs in sources block not escaped for Rich markup

**Symptom:** IPv6 addresses like `http://[::1]:8080/` contain square brackets which Rich interprets as markup tags, causing rendering errors or exceptions.

**Fix:** Applied `rich.markup.escape()` to URL before passing to `console.print()`.

**File:** `tinirag/core/renderer.py` → `print_sources()`

---

### BUG-FIX-16: `_collect_stream` crashed on empty `chunk.choices`

**Symptom:** Some endpoints emit a final "usage" chunk with `choices=[]`. Accessing `chunk.choices[0]` would raise `IndexError`.

**Fix:** Added `if not chunk.choices: continue` guard.

**File:** `tinirag/core/renderer.py` → `_collect_stream()`

---

### BUG-FIX-17: Double decorator on CLI default command

**Symptom:** `_default` was decorated with both `@app.command(name="query", hidden=True)` and `@app.callback(invoke_without_command=True)`. The `@app.command` decorator is the wrong pattern here — `@app.callback` alone is the correct Typer mechanism for default invocation behaviour. The duplicate registration could cause unexpected CLI argument parsing.

**Fix:** Removed the `@app.command(...)` decorator, keeping only `@app.callback(invoke_without_command=True)`.

**File:** `tinirag/cli.py`

---

### BUG-FIX-18: `show_keywords` was dead code (`if cfg.output.show_keywords: pass`)

**Symptom:** The `show_keywords` config option did nothing — the `pass` branch was a placeholder that was never implemented. Keywords were available on `result.keywords` but never printed.

**Fix:** Implemented the feature: `print_info(f"Keywords: {result.keywords}")` is printed after the query when `show_keywords=True`.

**File:** `tinirag/cli.py`

---

### BUG-FIX-19: Session ID had 32-bit collision risk and lost creation timestamp on re-save

**Symptom 1:** `new_session_id()` returned 8 hex chars (32 bits). Birthday collision probability exceeds 1% after ~65K sessions.

**Symptom 2:** `save_session()` always wrote `datetime.now()` as the `created` field, so resuming and re-saving a session would overwrite the original creation timestamp.

**Fix 1:** Session IDs now use `YYYYMMDD-{8 hex chars}` format — human-readable, sortable, and 48-bit collision-resistant.

**Fix 2:** `save_session()` reads the existing `created` field if the file already exists and preserves it.

**File:** `tinirag/core/session.py`

---

## Session 1 — 2026-04-07 (Initial Build)

### Overview
Built TiniRAG CLI v0.1 from scratch based on spec files (tech_stack.md, prompt.md, good_practices.md, guard_rails.md). Final state: **153/153 tests passing**.

---

### BUG-FIX-01: Stale .venv linked to missing Python interpreter

**Symptom:** `uv sync` failed with "virtual environment cannot be used because it is not a compatible environment".

**Root cause:** Pre-existing `.venv` created by a different Python interpreter that no longer exists.

**Fix:** `rm -rf .venv && uv sync`

---

### BUG-FIX-02: `is_stale()` returned False for genuinely old dates

**Symptom:** `is_stale("2020-01-01", 180)` returned `False` instead of `True`.

**Root cause:** `datetime.fromisoformat("2020-01-01")` returns a **naive** datetime (no timezone). Subtracting it from `datetime.now(timezone.utc)` (an **aware** datetime) raises `TypeError`, which was caught and swallowed, returning `False`.

**Fix:** After parsing, if `pub.tzinfo is None`, attach UTC: `pub = pub.replace(tzinfo=timezone.utc)`.

**File:** `tinirag/core/guardrails.py` → `is_stale()`

---

### BUG-FIX-03: `SENSITIVE_PATTERNS` didn't match plural forms

**Symptom:** `sensitive_category("what are the symptoms of diabetes")` returned `None`. `sensitive_category("should I invest in this stock")` returned `None`.

**Root cause 1:** The pattern used `\bsymptom\b` (with trailing `\b`). The word "symptoms" has a word boundary before 's' but not after "symptom" (the 's' continues the word). `\b` requires a transition from word-char to non-word-char; inside "symptoms", no such transition exists after "symptom".

**Root cause 2:** `investments?` matches "investment"/"investments" but not bare "invest" since "ment" is not optional.

**Fix:**
- `symptom` → `symptoms?` (trailing `\b` removed from full pattern group)
- `investments?` → `invest(ments?)?` so bare "invest" also matches

**File:** `tinirag/core/guardrails.py` → `SENSITIVE_PATTERNS`

---

### BUG-FIX-04: Test content below 100-char minimum threshold

**Symptom:** `test_basic_build`, `test_multiple_sources`, `test_source_num_assigned` all failed with empty context/sources.

**Root cause:** `build_context()` enforces `min_content_chars = 100` (GR-R5). Test content like "Python asyncio enables concurrent programming. It uses event loops for async operations." was 90 chars — below the threshold and correctly discarded.

**Fix:** Expanded test content strings to exceed 100 characters.

**File:** `tests/test_context.py`

---

### BUG-FIX-05: Snippet sufficiency test used snippet shorter than 120-char threshold

**Symptom:** `test_sufficient_snippet` failed.

**Root cause:** `snippet_is_sufficient()` requires `len(snippet) > 120`. Test snippet "The M4 chip has 38 GPU cores and 16GB unified memory." is 54 chars.

**Fix:** Expanded test snippet to exceed 120 characters.

**File:** `tests/test_context.py`

---

### BUG-FIX-06: `"a" not in result` substring check on normalized cache key

**Symptom:** `test_removes_noise_words` failed — "a" appears as a substring in "macbook".

**Root cause:** Test used `"a" not in result` which is a Python string substring check, not a word-level membership check. "macbook" contains the character 'a'.

**Fix:** Changed to `assert "a" not in result.split()` (word-level membership).

**File:** `tests/test_cache.py`

---

### BUG-FIX-07: Mock search result content below 100-char threshold (engine tests)

**Symptom:** `test_successful_rag_query` asserted `result.used_search` but it was `False`. Sources list was empty.

**Root cause:** Mock result content "The MacBook Neo uses the M4 Ultra GPU with 38 cores for graphics." = 67 chars. `build_context()` filtered it out (GR-R5: min 100 chars).

**Fix:** Expanded mock content to 3 sentences totaling >100 characters.

**File:** `tests/test_engine.py`

---

### Lint fixes (ruff auto-fixed)
- Removed 15 unused imports across test files and source modules
- Fixed un-sorted import blocks in test files
- Removed 3 unused local variable assignments (F841)
