"""Tests for tinirag.core.search (SearXNG integration, mocked HTTP)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tinirag.config import TiniRAGConfig
from tinirag.core.search import check_searxng, search_and_fetch


def _make_cfg() -> TiniRAGConfig:
    cfg = TiniRAGConfig()
    cfg.search.num_results = 3
    cfg.search.fetch_top_url = False  # disable URL fetching in unit tests
    return cfg


def _mock_searxng_response(results: list[dict]) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.headers = {"content-type": "application/json"}
    resp.json.return_value = {"results": results}
    return resp


class TestCheckSearxng:
    @pytest.mark.asyncio
    async def test_healthy_instance(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value = mock_client

            result = await check_searxng("http://localhost:8888")
        assert result is True

    @pytest.mark.asyncio
    async def test_unreachable_returns_false(self):
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(side_effect=Exception("Connection refused"))
            mock_client_cls.return_value = mock_client

            result = await check_searxng("http://localhost:8888")
        assert result is False


class TestSearchAndFetch:
    @pytest.mark.asyncio
    async def test_html_content_type_raises(self):
        """BUG-01: SearXNG returning HTML must raise RuntimeError."""
        cfg = _make_cfg()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "text/html; charset=utf-8"}
        mock_resp.json.return_value = {}

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value = mock_client

            with pytest.raises(RuntimeError, match="HTML instead of JSON"):
                await search_and_fetch("python asyncio", cfg)

    @pytest.mark.asyncio
    async def test_returns_results_with_content(self):
        cfg = _make_cfg()
        raw = [
            {
                "url": "https://docs.python.org/asyncio",
                "content": "Python asyncio provides event-driven concurrency primitives.",
                "title": "asyncio",
            }
        ]
        mock_resp = _mock_searxng_response(raw)

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value = mock_client

            results = await search_and_fetch("python asyncio", cfg)

        assert len(results) == 1
        assert results[0]["url"] == "https://docs.python.org/asyncio"
        assert results[0]["content"]

    @pytest.mark.asyncio
    async def test_blocked_domain_excluded(self):
        cfg = _make_cfg()
        raw = [
            {"url": "https://aicontentfa.com/article", "content": "AI generated content here."},
            {"url": "https://docs.python.org/asyncio", "content": "Python asyncio docs content."},
        ]
        mock_resp = _mock_searxng_response(raw)

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value = mock_client

            results = await search_and_fetch("python asyncio", cfg)

        urls = [r["url"] for r in results]
        assert not any("aicontentfa.com" in u for u in urls)

    @pytest.mark.asyncio
    async def test_empty_results_returned_as_empty_list(self):
        cfg = _make_cfg()
        mock_resp = _mock_searxng_response([])

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value = mock_client

            results = await search_and_fetch("python asyncio", cfg)

        assert results == []

    @pytest.mark.asyncio
    async def test_connection_error_raises(self):
        cfg = _make_cfg()

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(side_effect=Exception("Connection refused"))
            mock_client_cls.return_value = mock_client

            with pytest.raises(RuntimeError, match="unreachable"):
                await search_and_fetch("python asyncio", cfg)

    @pytest.mark.asyncio
    async def test_freshness_uses_configured_days(self):
        """GR-R2: freshness check must use cfg.guardrails.source_freshness_days, not hardcoded 180."""
        cfg = _make_cfg()
        cfg.guardrails.source_freshness_days = 30  # strict threshold

        # A result published ~60 days ago (stale under 30-day threshold)
        from datetime import datetime, timedelta, timezone

        stale_date = (datetime.now(timezone.utc) - timedelta(days=60)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        raw = [
            {
                "url": "https://example.com/article",
                "content": "Python asyncio content here.",
                "publishedDate": stale_date,
            }
        ]
        mock_resp = _mock_searxng_response(raw)

        with (
            patch("httpx.AsyncClient") as mock_client_cls,
            patch("tinirag.core.search.log_rail") as mock_log,
        ):
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value = mock_client

            await search_and_fetch("latest python asyncio", cfg, raw_query="latest python asyncio")

        # GR-R2 should have fired because 60 days > 30-day threshold
        rail_calls = [c for c in mock_log.call_args_list if c[0][0] == "GR-R2"]
        assert len(rail_calls) == 1
