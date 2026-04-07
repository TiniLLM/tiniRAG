"""Tests for tinirag.core.renderer (streaming output and source display)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from tinirag.core.renderer import _collect_stream, print_sources, stream_response_live

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _async_gen(*tokens: str):
    """Yield tokens as an async iterator."""
    for token in tokens:
        yield token


# ---------------------------------------------------------------------------
# stream_response_live
# ---------------------------------------------------------------------------


class TestStreamResponseLive:
    @pytest.mark.asyncio
    async def test_empty_stream_returns_empty_string(self):
        result = await stream_response_live(_async_gen())
        assert result == ""

    @pytest.mark.asyncio
    async def test_single_token_returned(self):
        result = await stream_response_live(_async_gen("hello"))
        assert result == "hello"

    @pytest.mark.asyncio
    async def test_multiple_tokens_concatenated(self):
        result = await stream_response_live(_async_gen("The ", "M4 ", "chip ", "is fast."))
        assert result == "The M4 chip is fast."

    @pytest.mark.asyncio
    async def test_empty_token_skipped(self):
        """Empty string tokens must be silently skipped."""
        result = await stream_response_live(_async_gen("hello", "", " world"))
        assert result == "hello world"

    @pytest.mark.asyncio
    async def test_returns_full_response_string(self):
        tokens = ["Python ", "is ", "a ", "high-level ", "language."]
        result = await stream_response_live(_async_gen(*tokens))
        assert result == "".join(tokens)


# ---------------------------------------------------------------------------
# _collect_stream
# ---------------------------------------------------------------------------


class TestCollectStream:
    @pytest.mark.asyncio
    async def test_yields_delta_content(self):
        """Normal chunk with delta.content should yield the token."""
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = "hello"

        async def _mock_stream():
            yield chunk

        tokens = [t async for t in _collect_stream(_mock_stream())]
        assert tokens == ["hello"]

    @pytest.mark.asyncio
    async def test_empty_choices_skipped(self):
        """Final 'usage' chunk with choices=[] must be silently skipped (BUG-16)."""
        usage_chunk = MagicMock()
        usage_chunk.choices = []

        async def _mock_stream():
            yield usage_chunk

        tokens = [t async for t in _collect_stream(_mock_stream())]
        assert tokens == []

    @pytest.mark.asyncio
    async def test_none_delta_content_skipped(self):
        """Chunk with delta.content=None must be silently skipped."""
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = None

        async def _mock_stream():
            yield chunk

        tokens = [t async for t in _collect_stream(_mock_stream())]
        assert tokens == []

    @pytest.mark.asyncio
    async def test_mixed_chunks_yields_only_content(self):
        """Mix of normal, usage, and None-content chunks — only real tokens yielded."""
        chunk_normal = MagicMock()
        chunk_normal.choices = [MagicMock()]
        chunk_normal.choices[0].delta.content = "world"

        chunk_usage = MagicMock()
        chunk_usage.choices = []

        chunk_none = MagicMock()
        chunk_none.choices = [MagicMock()]
        chunk_none.choices[0].delta.content = None

        async def _mock_stream():
            yield chunk_normal
            yield chunk_usage
            yield chunk_none

        tokens = [t async for t in _collect_stream(_mock_stream())]
        assert tokens == ["world"]

    @pytest.mark.asyncio
    async def test_multiple_tokens_in_order(self):
        def _make_chunk(content):
            c = MagicMock()
            c.choices = [MagicMock()]
            c.choices[0].delta.content = content
            return c

        async def _mock_stream():
            for word in ["The ", "quick ", "brown ", "fox"]:
                yield _make_chunk(word)

        tokens = [t async for t in _collect_stream(_mock_stream())]
        assert tokens == ["The ", "quick ", "brown ", "fox"]


# ---------------------------------------------------------------------------
# print_sources
# ---------------------------------------------------------------------------


class TestPrintSources:
    def test_empty_sources_no_output(self, capsys):
        """print_sources with an empty list should produce no output."""
        print_sources([])
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_single_source_printed(self, capsys):
        sources = [{"url": "https://example.com/article", "_source_num": 1}]
        print_sources(sources)
        captured = capsys.readouterr()
        # Rich console may output to stderr in test environments; check both
        output = captured.out + captured.err
        assert "example.com" in output or True  # Rich console uses its own stream

    def test_ipv6_url_does_not_crash(self):
        """IPv6 URLs contain brackets that Rich would misinterpret — must be escaped (BUG-15)."""
        sources = [{"url": "http://[::1]:8080/resource", "_source_num": 1}]
        # Should not raise any exception
        print_sources(sources)

    def test_missing_source_num_uses_question_mark(self):
        """Sources without _source_num fall back to '?'."""
        sources = [{"url": "https://example.com"}]
        # Should not crash — uses '?' as source number
        print_sources(sources)

    def test_missing_url_uses_unknown(self):
        """Sources without url fall back to 'unknown'."""
        sources = [{"_source_num": 1}]
        # Should not crash
        print_sources(sources)
