"""Tests for AuthConfig schemas and ScrapeRequest integration."""

import pytest
from pydantic import ValidationError

from schemas.scrape import AuthConfig, AuthType, ScrapeRequest, ScrapeResponse, ScrapeStatus


class TestAuthConfigSchema:
    def test_requires_credential_id(self) -> None:
        with pytest.raises(ValidationError):
            AuthConfig()  # type: ignore[call-arg]

    def test_defaults(self) -> None:
        cfg = AuthConfig(credential_id="cred-1")
        assert cfg.type is AuthType.FORM
        assert cfg.credential_id == "cred-1"
        assert cfg.login_url is None
        assert cfg.wait_timeout_ms == 15000

    def test_accepts_css_selectors(self) -> None:
        cfg = AuthConfig(
            credential_id="cred-1",
            username_selector='input[name="email"]',
            password_selector='input[type="password"]',
            submit_selector='button[type="submit"]',
        )
        assert cfg.username_selector == 'input[name="email"]'

    def test_accepts_all_auth_types(self) -> None:
        for t in ("form", "storage_state", "bearer", "basic"):
            cfg = AuthConfig(credential_id="cred-1", type=t)  # type: ignore[arg-type]
            assert cfg.type == t

    def test_rejects_unknown_auth_type(self) -> None:
        with pytest.raises(ValidationError):
            AuthConfig(credential_id="cred-1", type="totp")  # type: ignore[arg-type]

    def test_rejects_control_chars_in_credential_id(self) -> None:
        with pytest.raises(ValidationError):
            AuthConfig(credential_id="cred\x00id")

    def test_wait_timeout_bounds(self) -> None:
        with pytest.raises(ValidationError):
            AuthConfig(credential_id="cred-1", wait_timeout_ms=500)
        with pytest.raises(ValidationError):
            AuthConfig(credential_id="cred-1", wait_timeout_ms=200000)


class TestScrapeRequestAuthConfig:
    def test_auth_config_is_optional(self) -> None:
        req = ScrapeRequest(url="https://example.com")  # type: ignore[arg-type]
        assert req.auth_config is None

    def test_auth_config_accepted(self) -> None:
        req = ScrapeRequest(
            url="https://example.com",  # type: ignore[arg-type]
            auth_config=AuthConfig(credential_id="cred-1"),
        )
        assert req.auth_config is not None
        assert req.auth_config.credential_id == "cred-1"


class TestScrapeResponseStorageState:
    def test_storage_state_optional(self) -> None:
        resp = ScrapeResponse(
            url="https://example.com",
            status=ScrapeStatus.COMPLETED,
            scraper_used="PlaywrightScraper",
            processing_time_seconds=0.1,
        )
        assert resp.storage_state is None

    def test_storage_state_round_trips(self) -> None:
        state = {"cookies": [{"name": "sid", "value": "abc", "domain": ".x.com"}], "origins": []}
        resp = ScrapeResponse(
            url="https://example.com",
            status=ScrapeStatus.COMPLETED,
            scraper_used="PlaywrightScraper",
            processing_time_seconds=0.1,
            storage_state=state,
        )
        assert resp.storage_state == state


class TestServiceLayerAuthConfig:
    """The service-layer AuthConfig is a dataclass holding *resolved* credentials."""

    def test_dataclass_defaults(self) -> None:
        from services.base_scraper import AuthConfig as ServiceAuthConfig

        cfg = ServiceAuthConfig()
        assert cfg.type == "form"
        assert cfg.username == ""
        assert cfg.password == ""
        assert cfg.token == ""
        assert cfg.storage_state is None
        assert cfg.wait_timeout_ms == 15000

    def test_scrape_config_accepts_auth_config(self) -> None:
        from services.base_scraper import AuthConfig as ServiceAuthConfig
        from services.base_scraper import ScrapeConfig

        auth = ServiceAuthConfig(type="form", username="u", password="p")
        cfg = ScrapeConfig(auth_config=auth)
        assert cfg.auth_config is auth

    def test_scrape_result_storage_state(self) -> None:
        from services.base_scraper import ScrapeResult
        from services.base_scraper import ScrapeStatus as SStatus

        r = ScrapeResult(url="https://example.com", status=SStatus.COMPLETED)
        assert r.storage_state is None
        r.storage_state = {"cookies": []}
        assert r.storage_state == {"cookies": []}
