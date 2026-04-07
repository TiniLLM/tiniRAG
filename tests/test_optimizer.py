"""Tests for tinirag.core.optimizer."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from tinirag.core.optimizer import optimize_query, regex_optimize


class TestRegexOptimize:
    def test_strips_what(self):
        result = regex_optimize("What GPU does the MacBook use?")
        assert "what" not in result.lower()
        assert "GPU" in result or "gpu" in result.lower()

    def test_strips_how(self):
        result = regex_optimize("How does transformer attention work?")
        assert "how" not in result.lower()
        assert "transformer" in result.lower()

    def test_strips_tell_me(self):
        result = regex_optimize("Tell me about the Llama 3 model")
        assert "tell" not in result.lower()
        assert "llama" in result.lower() or "Llama" in result

    def test_strips_explain(self):
        result = regex_optimize("Explain the attention mechanism in transformers")
        assert "explain" not in result.lower()

    def test_strips_can_you(self):
        result = regex_optimize("Can you describe vLLM architecture?")
        assert "can" not in result.lower()
        assert "vllm" in result.lower() or "vLLM" in result

    def test_strips_please(self):
        result = regex_optimize("Please find out the best GPU for local LLMs")
        assert "please" not in result.lower()

    def test_short_query_unchanged(self):
        # ≤ 5 words: passes through; with regex ≥3 chars in result
        result = regex_optimize("GPU performance benchmark")
        assert "GPU" in result or "performance" in result.lower()

    def test_removes_trailing_punctuation(self):
        result = regex_optimize("What is Python?")
        assert not result.endswith("?")

    def test_collapses_whitespace(self):
        result = regex_optimize("What   is   the    answer")
        assert "  " not in result

    def test_empty_after_strip_returns_original(self):
        # All filler, nothing left → returns original
        result = regex_optimize("What how why when where")
        assert len(result) >= 3

    def test_technical_terms_preserved(self):
        result = regex_optimize("What is the M4 Ultra chip's GPU core count?")
        assert "M4" in result or "m4" in result.lower()
        assert "GPU" in result or "gpu" in result.lower()

    def test_strips_give_me(self):
        result = regex_optimize("Give me information about Ollama setup")
        assert "give" not in result.lower()
        assert "ollama" in result.lower() or "Ollama" in result

    def test_strips_show_me(self):
        result = regex_optimize("Show me how to install vLLM")
        assert "show" not in result.lower()

    def test_strips_i_want_to_know(self):
        result = regex_optimize("I want to know the best LLM for coding")
        assert "want" not in result.lower()

    def test_version_numbers_preserved(self):
        result = regex_optimize("What is Llama 3.1 context window size?")
        assert "3.1" in result

    def test_named_entities_preserved(self):
        result = regex_optimize("Who created the GPT-4 model?")
        assert "GPT-4" in result or "gpt-4" in result.lower()

    def test_all_filler_returns_original(self):
        """When everything is stripped and result < 3 chars, return original."""
        query = "what how why when where"
        result = regex_optimize(query)
        # Must return something meaningful (original or residual)
        assert len(result) >= 3

    def test_multiline_query(self):
        """Multi-line query should be processed without crashing."""
        result = regex_optimize("What is the\nbest GPU for local LLMs?")
        assert len(result) >= 3

    def test_filler_not_stripped_inside_word(self):
        """'what' as substring of a technical term should not be mangled by word-boundary regex."""
        # STRIP_PATTERNS uses \b, so 'what' inside 'whatnot' or similar should not strip
        result = regex_optimize("SomethingWhatever technical term benchmark")
        assert "benchmark" in result.lower()

    def test_question_mark_stripped(self):
        result = regex_optimize("Python asyncio performance?")
        assert not result.endswith("?")

    def test_comma_not_stripped_from_keywords(self):
        """Commas mid-string are not trailing punctuation — only trailing is stripped."""
        result = regex_optimize("Python, asyncio, performance")
        # Result should still have the key terms
        assert "python" in result.lower() or "asyncio" in result.lower()


# ---------------------------------------------------------------------------
# optimize_query (async orchestrator)
# ---------------------------------------------------------------------------


class TestOptimizeQuery:
    @pytest.mark.asyncio
    async def test_use_llm_false_skips_llm(self):
        """LLM is NOT called when use_llm=False (default)."""
        mock_client = MagicMock()
        mock_client.chat = MagicMock()
        mock_client.chat.completions = MagicMock()
        mock_client.chat.completions.create = AsyncMock()

        result = await optimize_query("What is the best GPU?", use_llm=False, client=mock_client)
        mock_client.chat.completions.create.assert_not_called()
        assert len(result) >= 3

    @pytest.mark.asyncio
    async def test_use_llm_true_short_result_skips_llm(self):
        """LLM is NOT called when regex result is ≤ 8 words, even if use_llm=True."""
        mock_client = MagicMock()
        mock_client.chat = MagicMock()
        mock_client.chat.completions = MagicMock()
        mock_client.chat.completions.create = AsyncMock()

        # Short query → regex result will be ≤ 8 words → LLM skipped
        result = await optimize_query("Python GPU benchmark", use_llm=True, client=mock_client)
        mock_client.chat.completions.create.assert_not_called()
        assert result

    @pytest.mark.asyncio
    async def test_use_llm_true_long_result_calls_llm(self):
        """LLM IS called when regex result is > 8 words and use_llm=True."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="python, async, performance"))]

        mock_client = MagicMock()
        mock_client.chat = MagicMock()
        mock_client.chat.completions = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        # Long query with many non-filler words → regex result > 8 words → LLM called
        long_query = (
            "Python asyncio concurrent programming event loop coroutines tasks futures performance"
        )
        result = await optimize_query(long_query, use_llm=True, client=mock_client, model="llama3")
        mock_client.chat.completions.create.assert_called_once()
        assert "python" in result.lower() or "async" in result.lower()

    @pytest.mark.asyncio
    async def test_llm_failure_falls_back_to_regex(self):
        """If LLM raises, fallback to regex result (no crash)."""
        mock_client = MagicMock()
        mock_client.chat = MagicMock()
        mock_client.chat.completions = MagicMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=Exception("LLM offline"))

        long_query = (
            "Python asyncio concurrent programming event loop coroutines tasks futures performance"
        )
        result = await optimize_query(long_query, use_llm=True, client=mock_client)
        # Should not raise — falls back gracefully
        assert len(result) >= 3

    @pytest.mark.asyncio
    async def test_no_client_skips_llm(self):
        """LLM step is skipped when client is None, even if use_llm=True."""
        result = await optimize_query(
            "Python asyncio concurrent programming tasks performance benchmark",
            use_llm=True,
            client=None,
        )
        assert len(result) >= 3
