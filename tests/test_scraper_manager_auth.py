"""Tests for ScraperManager auth behavior: force Playwright + bulk session reuse."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from services.base_scraper import AuthConfig, ScrapeConfig, ScrapeResult, ScraperType, ScrapeStatus
from services.scraper_manager import ScraperManager


@pytest.fixture
def manager() -> ScraperManager:
    return ScraperManager()


def _result(
    url: str,
    *,
    status: ScrapeStatus = ScrapeStatus.COMPLETED,
    storage_state: dict[str, Any] | None = None,
) -> ScrapeResult:
    return ScrapeResult(
        url=url,
        status=status,
        scraper_used="PlaywrightScraper",
        processing_time_seconds=0.1,
        storage_state=storage_state,
    )


class TestAuthForcesPlaywright:
    @pytest.mark.asyncio
    async def test_form_auth_forces_playwright_even_when_auto_requested(
        self, manager: ScraperManager
    ) -> None:
        """AUTO scraper_type must be overridden to Playwright when auth is form."""
        captured: dict[str, Any] = {}

        scraper = MagicMock()
        scraper.name = "PlaywrightScraper"
        scraper.scrape = AsyncMock(return_value=_result("https://x.com"))

        async def _get_scraper(stype: ScraperType, config: ScrapeConfig) -> Any:
            captured["scraper_type"] = stype
            return scraper

        with patch.object(manager, "_get_scraper", side_effect=_get_scraper):
            config = ScrapeConfig(auth_config=AuthConfig(type="form", username="u", password="p"))
            await manager.scrape_url("https://x.com", scraper_type=ScraperType.AUTO, config=config)

        assert captured["scraper_type"] == ScraperType.PLAYWRIGHT

    @pytest.mark.asyncio
    async def test_storage_state_auth_forces_playwright(self, manager: ScraperManager) -> None:
        captured: dict[str, Any] = {}

        scraper = MagicMock()
        scraper.name = "PlaywrightScraper"
        scraper.scrape = AsyncMock(return_value=_result("https://x.com"))

        async def _get_scraper(stype: ScraperType, config: ScrapeConfig) -> Any:
            captured["scraper_type"] = stype
            return scraper

        with patch.object(manager, "_get_scraper", side_effect=_get_scraper):
            config = ScrapeConfig(
                auth_config=AuthConfig(type="storage_state", storage_state={"cookies": []})
            )
            await manager.scrape_url("https://x.com", scraper_type=ScraperType.HTTPX, config=config)

        assert captured["scraper_type"] == ScraperType.PLAYWRIGHT

    @pytest.mark.asyncio
    async def test_bearer_auth_does_not_force_playwright(self, manager: ScraperManager) -> None:
        """Bearer/basic are just headers — any scraper can handle them."""
        captured: dict[str, Any] = {}

        scraper = MagicMock()
        scraper.name = "HttpxScraper"
        scraper.scrape = AsyncMock(return_value=_result("https://x.com"))

        async def _get_scraper(stype: ScraperType, config: ScrapeConfig) -> Any:
            captured["scraper_type"] = stype
            return scraper

        with patch.object(manager, "_get_scraper", side_effect=_get_scraper):
            config = ScrapeConfig(auth_config=AuthConfig(type="bearer", token="jwt"))
            await manager.scrape_url("https://x.com", scraper_type=ScraperType.HTTPX, config=config)

        assert captured["scraper_type"] == ScraperType.HTTPX


class TestBulkAuthReuse:
    @pytest.mark.asyncio
    async def test_form_auth_bulk_logs_in_once_reuses_storage_state(
        self, manager: ScraperManager
    ) -> None:
        """First URL logs in with form, subsequent URLs reuse storage_state."""
        captured_state = {"cookies": [{"name": "sid", "value": "xyz"}], "origins": []}
        scrape_calls: list[dict[str, Any]] = []

        async def fake_scrape_url(
            url: str,
            *,
            scraper_type: ScraperType = ScraperType.AUTO,
            config: ScrapeConfig | None = None,
            **_kw: Any,
        ) -> ScrapeResult:
            scrape_calls.append(
                {
                    "url": url,
                    "auth_type": config.auth_config.type if config and config.auth_config else None,
                    "storage_state": (
                        config.auth_config.storage_state if config and config.auth_config else None
                    ),
                }
            )
            return _result(url, storage_state=captured_state)

        urls = ["https://x.com/a", "https://x.com/b", "https://x.com/c"]
        config = ScrapeConfig(auth_config=AuthConfig(type="form", username="u", password="p"))

        with patch.object(manager, "scrape_url", side_effect=fake_scrape_url):
            results = await manager.scrape_bulk(urls, config=config)

        assert len(results) == 3
        assert all(r.status == ScrapeStatus.COMPLETED for r in results)
        # First call is form; subsequent are storage_state with captured state.
        assert scrape_calls[0]["auth_type"] == "form"
        assert scrape_calls[1]["auth_type"] == "storage_state"
        assert scrape_calls[1]["storage_state"] == captured_state
        assert scrape_calls[2]["auth_type"] == "storage_state"

    @pytest.mark.asyncio
    async def test_form_auth_bulk_first_login_failure_short_circuits(
        self, manager: ScraperManager
    ) -> None:
        """If the login URL fails, don't hammer the target with N more failed logins."""
        call_count = {"n": 0}

        async def fake_scrape_url(
            url: str,
            *,
            scraper_type: ScraperType = ScraperType.AUTO,
            config: ScrapeConfig | None = None,
            **_kw: Any,
        ) -> ScrapeResult:
            call_count["n"] += 1
            if call_count["n"] == 1:
                return _result(url, status=ScrapeStatus.FAILED)
            return _result(url)

        urls = ["https://x.com/a", "https://x.com/b", "https://x.com/c"]
        config = ScrapeConfig(auth_config=AuthConfig(type="form", username="u", password="p"))

        with patch.object(manager, "scrape_url", side_effect=fake_scrape_url):
            results = await manager.scrape_bulk(urls, config=config)

        # Only the first URL hit scrape_url; others short-circuit to FAILED.
        assert call_count["n"] == 1
        assert results[0].status == ScrapeStatus.FAILED
        assert results[1].status == ScrapeStatus.FAILED
        assert results[2].status == ScrapeStatus.FAILED
        assert "login" in (results[1].error or "").lower()

    @pytest.mark.asyncio
    async def test_bulk_without_auth_config_uses_existing_loop(
        self, manager: ScraperManager
    ) -> None:
        """No auth_config → existing per-URL loop (no storage-state swap)."""
        call_configs: list[ScrapeConfig | None] = []

        async def fake_scrape_url(
            url: str,
            *,
            scraper_type: ScraperType = ScraperType.AUTO,
            config: ScrapeConfig | None = None,
            **_kw: Any,
        ) -> ScrapeResult:
            call_configs.append(config)
            return _result(url)

        urls = ["https://x.com/a", "https://x.com/b"]
        with patch.object(manager, "scrape_url", side_effect=fake_scrape_url):
            results = await manager.scrape_bulk(urls, config=ScrapeConfig())

        assert len(results) == 2
        # No config mutation between calls.
        for cfg in call_configs:
            assert cfg is not None
            assert cfg.auth_config is None
