"""Tests for tinirag.core.model_detect (all HTTP mocked)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tinirag.core.model_detect import _pick_best, _reset_cache, detect_available_model


@pytest.fixture(autouse=True)
def reset_cache():
    """Clear the module-level model cache before each test."""
    _reset_cache()
    yield
    _reset_cache()


def _mock_models_response(model_ids: list[str]) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"data": [{"id": m} for m in model_ids]}
    return resp


class TestDetectAvailableModel:
    def test_empty_list_returns_none(self):
        with patch("httpx.get", return_value=_mock_models_response([])):
            result = detect_available_model("http://localhost:11434")
        assert result is None

    def test_single_model_returned(self):
        with patch("httpx.get", return_value=_mock_models_response(["llama3.2:3b"])):
            result = detect_available_model("http://localhost:11434")
        assert result == "llama3.2:3b"

    def test_multiple_models_prefers_instruct(self):
        models = ["llama3.2:3b", "qwen2.5:7b-instruct", "mistral:7b"]
        with patch("httpx.get", return_value=_mock_models_response(models)):
            result = detect_available_model("http://localhost:11434")
        assert result == "qwen2.5:7b-instruct"

    def test_prefers_smallest_eligible(self):
        models = ["llama3.1:70b-instruct", "llama3.2:3b-instruct", "qwen2.5:14b-instruct"]
        with patch("httpx.get", return_value=_mock_models_response(models)):
            result = detect_available_model("http://localhost:11434")
        assert result == "llama3.2:3b-instruct"

    def test_skips_models_under_3b(self):
        """Models smaller than 3b are too weak for RAG — skip them."""
        models = ["tinyllama:1.1b-instruct", "llama3.2:3b-instruct"]
        with patch("httpx.get", return_value=_mock_models_response(models)):
            result = detect_available_model("http://localhost:11434")
        assert result == "llama3.2:3b-instruct"

    def test_network_failure_returns_none(self):
        with patch("httpx.get", side_effect=Exception("Connection refused")):
            result = detect_available_model("http://localhost:11434")
        assert result is None

    def test_non_200_returns_none(self):
        resp = MagicMock()
        resp.status_code = 503
        with patch("httpx.get", return_value=resp):
            result = detect_available_model("http://localhost:11434")
        assert result is None

    def test_result_is_cached(self):
        models = ["llama3.2:3b"]
        with patch("httpx.get", return_value=_mock_models_response(models)) as mock_get:
            detect_available_model("http://localhost:11434")
            detect_available_model("http://localhost:11434")
        # Should only hit the network once
        assert mock_get.call_count == 1

    def test_calls_v1_models_endpoint(self):
        with patch("httpx.get", return_value=_mock_models_response(["llama3:8b"])) as mock_get:
            detect_available_model("http://localhost:11434")
        called_url = mock_get.call_args[0][0]
        assert called_url == "http://localhost:11434/v1/models"


class TestPickBest:
    def test_single_model(self):
        assert _pick_best(["llama3:8b"]) == "llama3:8b"

    def test_prefers_instruct_over_base(self):
        assert _pick_best(["llama3:8b", "llama3:8b-instruct"]) == "llama3:8b-instruct"

    def test_prefers_chat_marker(self):
        result = _pick_best(["mistral:7b", "mistral:7b-chat"])
        assert result == "mistral:7b-chat"

    def test_smallest_wins_among_eligible(self):
        models = ["qwen2.5:14b-instruct", "qwen2.5:7b-instruct", "qwen2.5:3b-instruct"]
        assert _pick_best(models) == "qwen2.5:3b-instruct"

    def test_sub_3b_fallback_when_all_small(self):
        """If only sub-3b models exist, fall back to first one."""
        models = ["tinyllama:1.1b-instruct", "phi:2b-instruct"]
        result = _pick_best(models)
        assert result == "tinyllama:1.1b-instruct"

    def test_no_instruct_uses_size_logic(self):
        models = ["llama3:70b", "llama3:8b", "llama3:3b"]
        assert _pick_best(models) == "llama3:3b"

    def test_no_size_in_name_uses_first(self):
        models = ["custom-model", "another-model"]
        assert _pick_best(models) == "custom-model"
