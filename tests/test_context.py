"""Tests for tinirag.core.context."""

from tinirag.core.context import (
    build_context,
    count_tokens,
    deduplicate_sources,
    model_context_window,
    root_domain,
    snippet_is_sufficient,
)


class TestCountTokens:
    def test_empty_string(self):
        assert count_tokens("", "llama3") == 0

    def test_local_model_no_crash(self):
        # BUG-02: must not raise KeyError for local model names
        count = count_tokens("Hello world", "llama3")
        assert count > 0

    def test_mistral_no_crash(self):
        count = count_tokens("Hello world", "mistral")
        assert count > 0

    def test_qwen_no_crash(self):
        count = count_tokens("The quick brown fox", "qwen2.5")
        assert count > 0

    def test_unknown_model_no_crash(self):
        count = count_tokens("Hello world", "totally-unknown-model-xyz")
        assert count > 0

    def test_longer_text_more_tokens(self):
        short = count_tokens("Hello", "llama3")
        long = count_tokens("Hello world, this is a longer sentence with more tokens.", "llama3")
        assert long > short

    def test_empty_model_name(self):
        # Should fall back gracefully
        count = count_tokens("Hello world", "")
        assert count > 0


class TestModelContextWindow:
    def test_llama3(self):
        assert model_context_window("llama3") == 8192

    def test_mistral(self):
        assert model_context_window("mistral-7b") == 4096

    def test_unknown_defaults_4096(self):
        assert model_context_window("unknown-model") == 4096

    def test_phi_small(self):
        assert model_context_window("phi-2") == 2048

    def test_case_insensitive(self):
        assert model_context_window("Llama3") == model_context_window("llama3")

    def test_llama31_not_confused_with_llama3(self):
        # Bug fix: "llama3" is a substring of "llama3.1" — must return 131072, not 8192
        assert model_context_window("llama3.1") == 131072
        assert model_context_window("llama3.1:8b") == 131072

    def test_llama3_still_correct(self):
        # Ensure llama3 still returns its own window after the ordering fix
        assert model_context_window("llama3") == 8192
        assert model_context_window("llama3:8b") == 8192


class TestRootDomain:
    def test_simple_domain(self):
        assert root_domain("https://example.com/page") == "example.com"

    def test_subdomain_stripped(self):
        assert root_domain("https://docs.example.com/api") == "example.com"

    def test_github(self):
        assert root_domain("https://github.com/org/repo") == "github.com"

    def test_empty_url(self):
        result = root_domain("")
        assert isinstance(result, str)


class TestDeduplicateSources:
    def test_removes_same_domain(self):
        results = [
            {"url": "https://example.com/page1", "content": "Content about GPUs."},
            {"url": "https://example.com/page2", "content": "More content about GPUs."},
        ]
        deduped = deduplicate_sources(results)
        assert len(deduped) == 1

    def test_keeps_different_domains(self):
        results = [
            {"url": "https://site-a.com/page", "content": "Content A about neural networks."},
            {"url": "https://site-b.com/page", "content": "Content B about deep learning."},
        ]
        deduped = deduplicate_sources(results)
        assert len(deduped) == 2

    def test_empty_list(self):
        assert deduplicate_sources([]) == []

    def test_similar_content_deduped(self):
        # Same content (>70% overlap) from different domains
        same = "The quick brown fox jumps. This is a long sentence about the same topic here."
        results = [
            {"url": "https://site-a.com", "content": same},
            {"url": "https://site-b.com", "content": same},
        ]
        deduped = deduplicate_sources(results, threshold=0.70)
        assert len(deduped) == 1


class TestBuildContext:
    def _make_result(self, url: str, content: str) -> dict:
        return {"url": url, "content": content, "snippet": content}

    def test_basic_build(self):
        results = [
            self._make_result(
                "https://example.com",
                "Python asyncio enables concurrent programming with coroutines. "
                "It uses an event loop to schedule and run async operations efficiently. "
                "The asyncio library is part of the Python standard library since version 3.4.",
            )
        ]
        ctx, sources = build_context(results, "python asyncio", "llama3")
        assert "[Source 1]" in ctx
        assert "example.com" in ctx
        assert len(sources) == 1

    def test_irrelevant_source_filtered(self):
        results = [
            self._make_result(
                "https://example.com",
                "Quantum computing uses qubits. Superposition enables parallel computation.",
            )
        ]
        # Query keywords don't appear in content
        ctx, sources = build_context(results, "macbook gpu chip", "llama3")
        assert len(sources) == 0

    def test_short_content_filtered(self):
        results = [
            self._make_result("https://example.com", "Short.")  # < 100 chars
        ]
        ctx, sources = build_context(results, "short content", "llama3")
        assert len(sources) == 0

    def test_multiple_sources(self):
        results = [
            self._make_result(
                "https://site-a.com",
                "Python asyncio is a library for async programming in Python applications. "
                "It enables writing concurrent code using async/await syntax without threads.",
            ),
            self._make_result(
                "https://site-b.com",
                "Python asyncio provides an event loop for concurrent coroutine execution. "
                "Multiple coroutines can run on a single thread using cooperative multitasking.",
            ),
        ]
        ctx, sources = build_context(results, "python asyncio", "llama3")
        assert len(sources) >= 1

    def test_source_num_assigned(self):
        results = [
            self._make_result(
                "https://example.com",
                "Python asyncio enables writing concurrent code using async and await syntax. "
                "It is included in the Python standard library and widely used for IO-bound tasks.",
            )
        ]
        ctx, sources = build_context(results, "python asyncio", "llama3")
        if sources:
            assert sources[0]["_source_num"] == 1


class TestSnippetIsSufficient:
    def test_sufficient_snippet(self):
        snippet = (
            "The M4 chip has 38 GPU cores and 16GB unified memory for high performance. "
            "Apple's M4 architecture delivers exceptional GPU performance compared to prior generations."
        )
        keywords = ["m4", "gpu", "cores"]
        assert snippet_is_sufficient(snippet, keywords)

    def test_short_snippet_not_sufficient(self):
        snippet = "M4 chip."  # too short
        keywords = ["m4", "chip"]
        assert not snippet_is_sufficient(snippet, keywords)

    def test_no_specifics_not_sufficient(self):
        snippet = "The new chip has many cores and lots of memory available for users."
        keywords = ["chip", "cores", "memory"]
        # Has keyword hits and long enough, but no numbers
        assert not snippet_is_sufficient(snippet, keywords)

    def test_empty_keywords(self):
        snippet = "The M4 chip has 38 GPU cores."
        assert not snippet_is_sufficient(snippet, [])

    def test_empty_snippet(self):
        assert not snippet_is_sufficient("", ["m4", "gpu"])

    def test_low_keyword_coverage(self):
        snippet = "The M4 chip has 38 GPU cores and 16GB unified memory storage."
        keywords = ["m4", "iphone", "camera", "zoom", "sensor"]
        # Only 1/5 keywords hit — below 70%
        assert not snippet_is_sufficient(snippet, keywords)
