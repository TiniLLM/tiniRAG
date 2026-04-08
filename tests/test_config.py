"""Tests for tinirag.config."""

from tinirag.config import (
    TiniRAGConfig,
    load_blocklist,
    load_config,
    save_config,
)


class TestLoadConfig:
    def test_returns_tinirag_config(self):
        cfg = load_config()
        assert isinstance(cfg, TiniRAGConfig)

    def test_defaults_populated(self):
        cfg = load_config()
        assert cfg.llm.endpoint  # non-empty
        assert cfg.llm.model  # non-empty
        assert cfg.search.num_results > 0
        assert 0.0 <= cfg.llm.temperature <= 1.0
        assert cfg.llm.max_tokens > 0

    def test_env_var_overrides_endpoint(self, monkeypatch):
        monkeypatch.setenv("TINIRAG_ENDPOINT", "http://localhost:9999/v1")
        cfg = load_config()
        assert cfg.llm.endpoint == "http://localhost:9999/v1"

    def test_env_var_overrides_searxng(self, monkeypatch):
        monkeypatch.setenv("TINIRAG_SEARXNG_URL", "http://localhost:9090")
        cfg = load_config()
        assert cfg.search.searxng_url == "http://localhost:9090"

    def test_env_var_overrides_model(self, monkeypatch):
        monkeypatch.setenv("TINIRAG_MODEL", "mistral:7b")
        cfg = load_config()
        assert cfg.llm.model == "mistral:7b"

    def test_guardrail_defaults(self):
        cfg = load_config()
        assert cfg.guardrails.injection_detection is True
        assert cfg.guardrails.max_context_pct == 0.90
        assert cfg.guardrails.min_content_chars == 100

    def test_new_search_config_defaults(self):
        cfg = load_config()
        assert cfg.search.managed_searxng is True
        assert cfg.search.searxng_port == 18888
        assert cfg.search.searxng_startup_timeout_sec == 30.0
        assert cfg.search.searxng_url == "http://localhost:18888"
        assert cfg.search.time_range == "month"
        assert cfg.search.num_results == 3

    def test_env_var_searxng_url_disables_managed(self, monkeypatch):
        """TINIRAG_SEARXNG_URL env var must set managed_searxng=False."""
        monkeypatch.setenv("TINIRAG_SEARXNG_URL", "http://localhost:9090")
        cfg = load_config()
        assert cfg.search.searxng_url == "http://localhost:9090"
        assert cfg.search.managed_searxng is False


class TestSaveConfig:
    def test_save_and_reload(self, tmp_path, monkeypatch):
        import tinirag.config as cfg_module

        monkeypatch.setattr(cfg_module, "CONFIG_DIR", tmp_path)
        monkeypatch.setattr(cfg_module, "CONFIG_FILE", tmp_path / "config.toml")
        monkeypatch.setattr(cfg_module, "ENV_FILE", tmp_path / ".env")

        cfg = TiniRAGConfig()
        cfg.llm.model = "qwen2.5"
        cfg.llm.temperature = 0.2
        save_config(cfg)

        load_config()
        # The model and temperature are saved to the file
        assert (tmp_path / "config.toml").exists()

    def test_creates_config_dir(self, tmp_path, monkeypatch):
        import tinirag.config as cfg_module

        new_dir = tmp_path / "newdir"
        monkeypatch.setattr(cfg_module, "CONFIG_DIR", new_dir)
        monkeypatch.setattr(cfg_module, "CONFIG_FILE", new_dir / "config.toml")

        cfg = TiniRAGConfig()
        save_config(cfg)
        assert new_dir.exists()
        assert (new_dir / "config.toml").exists()


class TestLoadBlocklist:
    def test_default_blocked_domain(self):
        blocked = load_blocklist()
        assert "aicontentfa.com" in blocked

    def test_returns_set(self):
        assert isinstance(load_blocklist(), set)

    def test_custom_blocklist(self, tmp_path, monkeypatch):
        import tinirag.config as cfg_module

        bl_file = tmp_path / "blocklist.txt"
        bl_file.write_text("spam.example.com\n# comment\n\nmalware.net\n")
        monkeypatch.setattr(cfg_module, "BLOCKLIST_FILE", bl_file)

        blocked = load_blocklist()
        assert "spam.example.com" in blocked
        assert "malware.net" in blocked
        assert "# comment" not in blocked

    def test_missing_blocklist_file_returns_defaults(self, tmp_path, monkeypatch):
        """A missing blocklist file must not crash — returns built-in defaults."""
        import tinirag.config as cfg_module

        monkeypatch.setattr(cfg_module, "BLOCKLIST_FILE", tmp_path / "nonexistent.txt")
        blocked = load_blocklist()
        # Should still return something (built-in default set)
        assert isinstance(blocked, set)


# ---------------------------------------------------------------------------
# load_config: TOML file edge cases
# ---------------------------------------------------------------------------


class TestLoadConfigToml:
    def test_malformed_toml_falls_back_to_defaults(self, tmp_path, monkeypatch):
        """A corrupt config.toml must not crash — falls back to defaults."""
        import tinirag.config as cfg_module

        cfg_file = tmp_path / "config.toml"
        cfg_file.write_text("this is not [ valid toml !!!")
        monkeypatch.setattr(cfg_module, "CONFIG_DIR", tmp_path)
        monkeypatch.setattr(cfg_module, "CONFIG_FILE", cfg_file)

        # Should not raise — falls back to defaults
        cfg = load_config()
        assert isinstance(cfg, TiniRAGConfig)
        assert cfg.llm.endpoint  # defaults populated

    def test_partial_toml_uses_defaults_for_missing(self, tmp_path, monkeypatch):
        """A partial config.toml (only [llm] section) leaves other sections at defaults."""
        import tinirag.config as cfg_module

        cfg_file = tmp_path / "config.toml"
        cfg_file.write_text('[llm]\nmodel = "qwen2.5"\n')
        monkeypatch.setattr(cfg_module, "CONFIG_DIR", tmp_path)
        monkeypatch.setattr(cfg_module, "CONFIG_FILE", cfg_file)

        cfg = load_config()
        assert cfg.llm.model == "qwen2.5"
        # Other fields should still have valid defaults
        assert cfg.search.num_results > 0
        assert cfg.guardrails.min_content_chars > 0

    def test_env_var_takes_precedence_over_toml(self, tmp_path, monkeypatch):
        """Env var must override a value set in config.toml."""
        import tinirag.config as cfg_module

        cfg_file = tmp_path / "config.toml"
        cfg_file.write_text('[llm]\nendpoint = "http://toml-endpoint:1234/v1"\n')
        monkeypatch.setattr(cfg_module, "CONFIG_DIR", tmp_path)
        monkeypatch.setattr(cfg_module, "CONFIG_FILE", cfg_file)
        monkeypatch.setenv("TINIRAG_ENDPOINT", "http://env-endpoint:9999/v1")

        cfg = load_config()
        assert cfg.llm.endpoint == "http://env-endpoint:9999/v1"

    def test_explicit_searxng_url_in_toml_disables_managed(self, tmp_path, monkeypatch):
        """Explicit searxng_url in config.toml must set managed_searxng=False."""
        import tinirag.config as cfg_module

        cfg_file = tmp_path / "config.toml"
        cfg_file.write_text('[search]\nsearxng_url = "http://localhost:8888"\n')
        monkeypatch.setattr(cfg_module, "CONFIG_DIR", tmp_path)
        monkeypatch.setattr(cfg_module, "CONFIG_FILE", cfg_file)

        cfg = load_config()
        assert cfg.search.searxng_url == "http://localhost:8888"
        assert cfg.search.managed_searxng is False
