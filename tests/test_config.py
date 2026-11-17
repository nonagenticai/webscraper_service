"""Tests for configuration settings."""

import os
from unittest.mock import patch

from config import Settings, get_config, get_settings


class TestSettings:
    """Tests for Settings class."""

    def test_default_settings(self, monkeypatch):
        """Test default setting values.

        Every assertion here is about a DEFAULT, so the ambient environment has
        to be cleared first or the test measures the runner instead of the code.
        `environment` is the field that actually bit: it was only ever asserted
        to be "development" because ENVIRONMENT happened to be unset, and when
        gitops #2432 started exporting ENVIRONMENT=test to every python-service
        consumer this went red on main with no change in this repo. Clearing the
        var makes the test assert what its name claims — the declared default —
        rather than whatever the CI archetype exports today.
        """
        monkeypatch.delenv("ENVIRONMENT", raising=False)
        settings = Settings()

        assert settings.service_name == "webscraper-service"
        assert settings.version == "1.0.0"
        assert settings.environment == "development"
        assert settings.host == "0.0.0.0"
        assert settings.port == 9119
        assert settings.log_level == "INFO"
        assert settings.playwright_headless is True
        assert settings.scraping_default_delay == 1.0
        assert settings.scraping_max_concurrent == 16
        assert settings.scrapy_crawl_timeout == 120.0
        assert settings.scrapy_max_pages == 50
        # No hardcoded fallbacks for upstream service URLs.
        assert settings.secrets_service_url == ""
        assert settings.license_introspect_url == ""

    def test_log_level_resolves_to_logging_constant(self):
        """main.py wires basicConfig via getattr(logging, log_level.upper()):
        the configured value must resolve to a real logging level (was hardcoded
        INFO, so LOG_LEVEL had no effect)."""
        import logging

        assert getattr(logging, Settings().log_level.upper(), logging.INFO) == logging.INFO
        with patch.dict(os.environ, {"LOG_LEVEL": "WARNING"}, clear=False):
            assert getattr(logging, Settings().log_level.upper(), logging.INFO) == logging.WARNING

    def test_settings_from_env(self):
        """Test settings loaded from environment variables."""
        env_vars = {
            "SERVICE_NAME": "custom-service",
            "VERSION": "2.0.0",
            "ENVIRONMENT": "production",
            "HOST": "127.0.0.1",
            "PORT": "8080",
            "LOG_LEVEL": "DEBUG",
            "PLAYWRIGHT_HEADLESS": "false",
            "SCRAPING_DEFAULT_DELAY": "2.0",
            "SCRAPING_USER_AGENT": "CustomBot/1.0",
            "SCRAPING_MAX_CONCURRENT": "32",
            # Required in production (no hardcoded fallbacks).
            "SECRETS_SERVICE_URL": "http://secrets.test.svc:8543",
            "LICENSE_INTROSPECT_URL": "http://licenses.test.svc:8544/license/introspect",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            # Clear the cache to get fresh settings
            get_settings.cache_clear()
            settings = Settings()

            assert settings.service_name == "custom-service"
            assert settings.version == "2.0.0"
            assert settings.environment == "production"
            assert settings.host == "127.0.0.1"
            assert settings.port == 8080
            assert settings.log_level == "DEBUG"
            assert settings.playwright_headless is False
            assert settings.scraping_default_delay == 2.0
            assert settings.scraping_user_agent == "CustomBot/1.0"
            assert settings.scraping_max_concurrent == 32

        # Restore cache
        get_settings.cache_clear()


class TestGetSettings:
    """Tests for get_settings function."""

    def test_get_settings_returns_settings(self):
        """Test that get_settings returns a Settings instance."""
        get_settings.cache_clear()
        settings = get_settings()
        assert isinstance(settings, Settings)

    def test_get_settings_cached(self):
        """Test that get_settings returns cached instance."""
        get_settings.cache_clear()
        settings1 = get_settings()
        settings2 = get_settings()
        assert settings1 is settings2

    def test_get_config_alias(self):
        """Test that get_config is an alias for get_settings."""
        get_settings.cache_clear()
        _settings = get_settings()
        config = get_config()
        # Both should return Settings instances (but not necessarily same instance
        # since get_config calls get_settings which is cached)
        assert isinstance(config, Settings)


class TestSettingsValidation:
    """Tests for settings validation."""

    def test_port_as_string(self):
        """Test that port can be set as string and converted."""
        with patch.dict(os.environ, {"PORT": "9999"}, clear=False):
            settings = Settings()
            assert settings.port == 9999
            assert isinstance(settings.port, int)

    def test_boolean_settings(self):
        """Test boolean settings parsing."""
        test_cases = [
            ("true", True),
            ("True", True),
            ("TRUE", True),
            ("1", True),
            ("false", False),
            ("False", False),
            ("FALSE", False),
            ("0", False),
        ]

        for env_value, expected in test_cases:
            with patch.dict(os.environ, {"PLAYWRIGHT_HEADLESS": env_value}, clear=False):
                settings = Settings()
                assert settings.playwright_headless is expected, f"Failed for {env_value}"

    def test_float_settings(self):
        """Test float settings parsing."""
        with patch.dict(os.environ, {"SCRAPING_DEFAULT_DELAY": "45.5"}, clear=False):
            settings = Settings()
            assert settings.scraping_default_delay == 45.5
            assert isinstance(settings.scraping_default_delay, float)


class TestProductionValidator:
    """Tests for the _REQUIRED_IN_PRODUCTION runtime validator."""

    def test_production_raises_when_required_missing(self):
        """Production env without required URL fields must raise RuntimeError."""
        import pytest

        env = os.environ.copy()
        env["ENVIRONMENT"] = "production"
        env.pop("SECRETS_SERVICE_URL", None)
        env.pop("LICENSE_INTROSPECT_URL", None)
        with patch.dict(os.environ, env, clear=True):
            get_settings.cache_clear()
            with pytest.raises(RuntimeError, match="missing required fields"):
                Settings()

    def test_non_production_allows_empty(self):
        """Non-production env keeps empty defaults without raising."""
        env = {"ENVIRONMENT": "test"}
        with patch.dict(os.environ, env, clear=True):
            get_settings.cache_clear()
            settings = Settings()
            assert settings.environment == "test"
            assert settings.secrets_service_url == ""


class TestSettingsDefaults:
    """Tests for default value behaviors."""

    def test_extra_env_vars_ignored(self):
        """Test that extra environment variables are ignored."""
        with patch.dict(os.environ, {"UNKNOWN_SETTING": "value"}, clear=False):
            # Should not raise
            settings = Settings()
            assert not hasattr(settings, "unknown_setting")
