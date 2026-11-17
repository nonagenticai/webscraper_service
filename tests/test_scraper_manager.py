"""Tests for ScraperManager."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from services.base_scraper import (
    AuthConfig,
    ScrapeConfig,
    ScrapeResult,
    ScraperType,
    ScrapeStatus,
)
from services.scraper_manager import ScraperManager, get_scraper_manager


@pytest.fixture
def scraper_manager():
    """Create a fresh ScraperManager instance."""
    return ScraperManager()


@pytest.fixture
def mock_scrape_result():
    """Create a mock scrape result."""
    return ScrapeResult(
        url="https://example.com",
        html_content="<html><body>Test</body></html>",
        text_content="Test",
        title="Test Page",
        metadata={"status_code": 200},
        links=[],
        images=[],
        status=ScrapeStatus.COMPLETED,
        scraper_used="PlaywrightScraper",
        processing_time_seconds=1.0,
    )


class TestScraperManager:
    """Tests for ScraperManager class."""

    @pytest.mark.asyncio
    async def test_scrape_url_success(self, scraper_manager, mock_scrape_result):
        mock_playwright = MagicMock()
        mock_playwright.scrape = AsyncMock(return_value=mock_scrape_result)
        mock_playwright.name = "PlaywrightScraper"

        with patch.object(scraper_manager, "_get_scraper", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_playwright

            result = await scraper_manager.scrape_url(
                "https://example.com", scraper_type=ScraperType.PLAYWRIGHT
            )

            assert result.status == ScrapeStatus.COMPLETED
            assert result.url == "https://example.com"
            mock_playwright.scrape.assert_called_once()

    @pytest.mark.asyncio
    async def test_scrape_url_with_config(self, scraper_manager, mock_scrape_result):
        mock_playwright = MagicMock()
        mock_playwright.scrape = AsyncMock(return_value=mock_scrape_result)
        mock_playwright.name = "PlaywrightScraper"

        config = ScrapeConfig(timeout=60, screenshot=True, user_agent="TestAgent")

        with patch.object(scraper_manager, "_get_scraper", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_playwright

            result = await scraper_manager.scrape_url(
                "https://example.com",
                scraper_type=ScraperType.PLAYWRIGHT,
                config=config,
                timeout=90,
                screenshot=True,
                headers={"X-Test": "value"},
                cookies={"session": "abc"},
                user_agent="OverrideAgent",
            )

            assert result.status == ScrapeStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_scrape_url_error_handling(self, scraper_manager):
        mock_playwright = MagicMock()
        mock_playwright.scrape = AsyncMock(side_effect=Exception("Network error"))
        mock_playwright.name = "PlaywrightScraper"

        with patch.object(scraper_manager, "_get_scraper", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_playwright

            result = await scraper_manager.scrape_url(
                "https://example.com", scraper_type=ScraperType.PLAYWRIGHT
            )

            assert result.status == ScrapeStatus.FAILED
            assert "Network error" in result.error

    @pytest.mark.asyncio
    async def test_scrape_bulk_success(self, scraper_manager, mock_scrape_result):
        mock_playwright = MagicMock()
        mock_playwright.scrape = AsyncMock(return_value=mock_scrape_result)
        mock_playwright.name = "PlaywrightScraper"

        with patch.object(scraper_manager, "_get_scraper", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_playwright

            urls = ["https://example.com", "https://example.org"]
            results = await scraper_manager.scrape_bulk(urls, scraper_type=ScraperType.PLAYWRIGHT)

            assert len(results) == 2
            assert all(r.status == ScrapeStatus.COMPLETED for r in results)

    @pytest.mark.asyncio
    async def test_scrape_bulk_partial_failure(self, scraper_manager, mock_scrape_result):
        call_count = 0

        async def mock_scrape_url(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                return ScrapeResult(
                    url=args[0] if args else "https://fail.com",
                    status=ScrapeStatus.FAILED,
                    error="Failed",
                    scraper_used="unknown",
                )
            return mock_scrape_result

        with patch.object(scraper_manager, "scrape_url", side_effect=mock_scrape_url):
            urls = ["https://example.com", "https://fail.com", "https://example.org"]
            results = await scraper_manager.scrape_bulk(urls)

            assert len(results) == 3
            completed = [r for r in results if r.status == ScrapeStatus.COMPLETED]
            failed = [r for r in results if r.status == ScrapeStatus.FAILED]
            assert len(completed) == 2
            assert len(failed) == 1

    @pytest.mark.asyncio
    async def test_scrape_bulk_exception_in_loop(self, scraper_manager):
        async def mock_scrape(*args, **kwargs):
            raise Exception("Unexpected error")

        with patch.object(scraper_manager, "scrape_url", side_effect=mock_scrape):
            urls = ["https://example.com"]
            results = await scraper_manager.scrape_bulk(urls)

            assert len(results) == 1
            assert results[0].status == ScrapeStatus.FAILED
            assert "Unexpected error" in results[0].error

    @pytest.mark.asyncio
    async def test_get_scraper_creates_playwright(self, scraper_manager):
        with patch("services.scraper_manager.PlaywrightScraper") as mock_playwright:
            mock_instance = MagicMock()
            mock_playwright.return_value = mock_instance

            scraper = await scraper_manager._get_scraper(ScraperType.PLAYWRIGHT)

            assert scraper == mock_instance
            mock_playwright.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_scraper_creates_httpx(self, scraper_manager):
        with patch("services.scraper_manager.HttpxScraper") as mock_httpx:
            mock_instance = MagicMock()
            mock_httpx.return_value = mock_instance

            scraper = await scraper_manager._get_scraper(ScraperType.HTTPX)

            assert scraper == mock_instance
            mock_httpx.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_scraper_creates_scrapy(self, scraper_manager):
        with patch("services.scraper_manager.ScrapyScraper") as mock_scrapy:
            mock_instance = MagicMock()
            mock_scrapy.return_value = mock_instance

            scraper = await scraper_manager._get_scraper(ScraperType.SCRAPY)

            assert scraper == mock_instance
            mock_scrapy.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_scraper_reuses_existing_playwright(self, scraper_manager):
        # Reuses the cached instance and DOES NOT mutate its self.config — the
        # per-request config travels with each scrape() call instead, so a
        # concurrent request can't stomp another request's auth/cookies.
        original = ScrapeConfig()
        mock_playwright = MagicMock()
        mock_playwright.config = original
        scraper_manager._playwright_scraper = mock_playwright

        config = ScrapeConfig(timeout=90)
        scraper = await scraper_manager._get_scraper(ScraperType.PLAYWRIGHT, config)

        assert scraper == mock_playwright
        assert scraper.config is original
        assert scraper.config.timeout != 90

    @pytest.mark.asyncio
    async def test_get_scraper_reuses_existing_httpx(self, scraper_manager):
        original = ScrapeConfig()
        mock_httpx = MagicMock()
        mock_httpx.config = original
        scraper_manager._httpx_scraper = mock_httpx

        config = ScrapeConfig(timeout=90)
        scraper = await scraper_manager._get_scraper(ScraperType.HTTPX, config)

        assert scraper == mock_httpx
        assert scraper.config is original
        assert scraper.config.timeout != 90

    @pytest.mark.asyncio
    async def test_get_scraper_reuses_existing_scrapy(self, scraper_manager):
        original = ScrapeConfig()
        mock_scrapy = MagicMock()
        mock_scrapy.config = original
        scraper_manager._scrapy_scraper = mock_scrapy

        config = ScrapeConfig(timeout=90)
        scraper = await scraper_manager._get_scraper(ScraperType.SCRAPY, config)

        assert scraper == mock_scrapy
        assert scraper.config is original
        assert scraper.config.timeout != 90

    @pytest.mark.asyncio
    async def test_scrape_url_passes_config_and_timeout_to_scraper(
        self, scraper_manager, mock_scrape_result
    ):
        # The resolved per-request config (and timeout) must reach scraper.scrape
        # as kwargs — that is how request-local state travels without mutating the
        # shared instance.
        mock_playwright = MagicMock()
        mock_playwright.scrape = AsyncMock(return_value=mock_scrape_result)
        mock_playwright.name = "PlaywrightScraper"

        config = ScrapeConfig(timeout=45, user_agent="Probe")
        with patch.object(scraper_manager, "_get_scraper", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_playwright
            await scraper_manager.scrape_url(
                "https://example.com", scraper_type=ScraperType.PLAYWRIGHT, config=config
            )

        _, call_kwargs = mock_playwright.scrape.call_args
        assert call_kwargs["config"] is config
        assert call_kwargs["timeout"] == 45

    @pytest.mark.asyncio
    async def test_concurrent_requests_do_not_bleed_config(self, scraper_manager):
        # Regression: two concurrent scrape_url calls share ONE scraper instance,
        # but each scrape() must see its OWN config (auth/cookies) — no cross-bleed.
        class _RecordingScraper:
            name = "RecordingScraper"

            def __init__(self) -> None:
                self.config = ScrapeConfig()  # shared singleton state
                self.seen: list[ScrapeConfig] = []

            async def scrape(self, url: str, **kwargs: object) -> ScrapeResult:
                cfg = kwargs.get("config")
                await asyncio.sleep(0)  # force interleave between the two calls
                assert isinstance(cfg, ScrapeConfig)
                self.seen.append(cfg)
                return ScrapeResult(url=url, status=ScrapeStatus.COMPLETED, scraper_used=self.name)

        shared = _RecordingScraper()
        cfg_a = ScrapeConfig(
            cookies={"who": "A"}, auth_config=AuthConfig(type="bearer", token="tok-A")
        )
        cfg_b = ScrapeConfig(
            cookies={"who": "B"}, auth_config=AuthConfig(type="bearer", token="tok-B")
        )

        with patch.object(scraper_manager, "_get_scraper", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = shared
            # Loopback hosts are allowlisted in the test env so the SSRF guard
            # (which runs before _get_scraper) doesn't reject these before the
            # mocked scraper records the per-request config.
            await asyncio.gather(
                scraper_manager.scrape_url(
                    "http://127.0.0.1/a", scraper_type=ScraperType.PLAYWRIGHT, config=cfg_a
                ),
                scraper_manager.scrape_url(
                    "http://127.0.0.1/b", scraper_type=ScraperType.PLAYWRIGHT, config=cfg_b
                ),
            )

        seen_owners = {c.cookies["who"] for c in shared.seen}
        seen_tokens = {c.auth_config.token for c in shared.seen if c.auth_config}
        assert seen_owners == {"A", "B"}
        assert seen_tokens == {"tok-A", "tok-B"}

    @pytest.mark.asyncio
    async def test_close_scrapers(self, scraper_manager):
        mock_playwright = MagicMock()
        mock_playwright.close = AsyncMock()
        scraper_manager._playwright_scraper = mock_playwright

        mock_httpx = MagicMock()
        mock_httpx.close = AsyncMock()
        scraper_manager._httpx_scraper = mock_httpx

        mock_scrapy = MagicMock()
        mock_scrapy.close = AsyncMock()
        scraper_manager._scrapy_scraper = mock_scrapy

        await scraper_manager.close()

        mock_playwright.close.assert_called_once()
        mock_httpx.close.assert_called_once()
        mock_scrapy.close.assert_called_once()
        assert scraper_manager._playwright_scraper is None
        assert scraper_manager._httpx_scraper is None
        assert scraper_manager._scrapy_scraper is None

    @pytest.mark.asyncio
    async def test_close_when_no_scrapers(self, scraper_manager):
        await scraper_manager.close()


class TestAutoSelection:
    """Tests for AUTO mode scraper selection."""

    @pytest.mark.asyncio
    async def test_auto_follow_links_uses_scrapy(self, scraper_manager):
        """AUTO with follow_links=True should select ScrapyScraper."""
        config = ScrapeConfig(follow_links=True, max_depth=2)
        scraper = await scraper_manager._select_auto_scraper("https://example.com", config)
        from services.scrapy_scraper import ScrapyScraper

        assert isinstance(scraper, ScrapyScraper)

    @pytest.mark.asyncio
    async def test_auto_max_depth_uses_scrapy(self, scraper_manager):
        """AUTO with max_depth > 1 should select ScrapyScraper."""
        config = ScrapeConfig(max_depth=3)
        scraper = await scraper_manager._select_auto_scraper("https://example.com", config)
        from services.scrapy_scraper import ScrapyScraper

        assert isinstance(scraper, ScrapyScraper)

    @pytest.mark.asyncio
    async def test_auto_js_enabled_uses_playwright(self, scraper_manager):
        """AUTO with javascript_enabled=True should select PlaywrightScraper."""
        config = ScrapeConfig(javascript_enabled=True)
        scraper = await scraper_manager._select_auto_scraper("https://example.com", config)
        from services.playwright_scraper import PlaywrightScraper

        assert isinstance(scraper, PlaywrightScraper)

    @pytest.mark.asyncio
    async def test_auto_wait_for_selector_forces_playwright(self, scraper_manager):
        """AUTO + wait_for_selector must select Playwright even on a HEAD-200 page.

        wait_for_selector is a browser-only signal (httpx does a single GET and
        cannot honour it), so routing to httpx would silently drop the wait and
        return pre-hydration content reported as COMPLETED. HEAD is patched to 200
        to prove the selection is forced BEFORE the HEAD short-circuit to httpx.
        """
        config = ScrapeConfig()
        with patch.object(scraper_manager, "_head_check", return_value=200) as head:
            scraper = await scraper_manager._select_auto_scraper(
                "https://example.com", config, wait_for_selector="#app .loaded"
            )
        from services.playwright_scraper import PlaywrightScraper

        assert isinstance(scraper, PlaywrightScraper)
        # Forced by the browser-only signal — HEAD pre-check never runs.
        head.assert_not_called()

    @pytest.mark.asyncio
    async def test_auto_non_default_wait_until_forces_playwright(self, scraper_manager):
        """AUTO + non-default wait_until (e.g. networkidle) must select Playwright."""
        config = ScrapeConfig()
        with patch.object(scraper_manager, "_head_check", return_value=200):
            scraper = await scraper_manager._select_auto_scraper(
                "https://example.com", config, wait_until="networkidle"
            )
        from services.playwright_scraper import PlaywrightScraper

        assert isinstance(scraper, PlaywrightScraper)

    @pytest.mark.asyncio
    async def test_auto_default_wait_until_still_uses_httpx(self, scraper_manager):
        """The default wait_until='load' is NOT browser-forcing — httpx still wins
        on a HEAD-200 page (guards against over-forcing every AUTO request)."""
        config = ScrapeConfig()
        with patch.object(scraper_manager, "_head_check", return_value=200):
            scraper = await scraper_manager._select_auto_scraper(
                "https://example.com", config, wait_until="load"
            )
        from services.httpx_scraper import HttpxScraper

        assert isinstance(scraper, HttpxScraper)

    @pytest.mark.asyncio
    async def test_auto_head_200_uses_httpx(self, scraper_manager):
        """AUTO with HEAD returning 200 should select HttpxScraper."""
        config = ScrapeConfig()
        with patch.object(scraper_manager, "_head_check", return_value=200):
            scraper = await scraper_manager._select_auto_scraper("https://example.com", config)
        from services.httpx_scraper import HttpxScraper

        assert isinstance(scraper, HttpxScraper)

    @pytest.mark.asyncio
    async def test_auto_head_403_uses_curl_cffi(self, scraper_manager):
        """AUTO with HEAD 403 skips httpx and tries curl_cffi (Chrome impersonation)
        first — cheaper than a browser; scrape_url escalates to Playwright if it
        still FAILs."""
        config = ScrapeConfig()
        with patch.object(scraper_manager, "_head_check", return_value=403):
            scraper = await scraper_manager._select_auto_scraper(
                "https://blocked.example.com", config
            )
        from services.curl_cffi_scraper import CurlCffiScraper

        assert isinstance(scraper, CurlCffiScraper)

    @pytest.mark.asyncio
    async def test_auto_head_429_uses_curl_cffi(self, scraper_manager):
        """AUTO with HEAD 429 skips httpx and tries curl_cffi first."""
        config = ScrapeConfig()
        with patch.object(scraper_manager, "_head_check", return_value=429):
            scraper = await scraper_manager._select_auto_scraper(
                "https://ratelimited.example.com", config
            )
        from services.curl_cffi_scraper import CurlCffiScraper

        assert isinstance(scraper, CurlCffiScraper)

    @pytest.mark.asyncio
    async def test_auto_head_timeout_uses_playwright(self, scraper_manager):
        """AUTO with HEAD timeout should use PlaywrightScraper."""
        config = ScrapeConfig()
        with patch.object(scraper_manager, "_head_check", return_value=None):
            scraper = await scraper_manager._select_auto_scraper(
                "https://timeout.example.com", config
            )
        from services.playwright_scraper import PlaywrightScraper

        assert isinstance(scraper, PlaywrightScraper)

    @pytest.mark.asyncio
    async def test_auto_head_500_uses_playwright(self, scraper_manager):
        """AUTO with HEAD returning 500 should use PlaywrightScraper."""
        config = ScrapeConfig()
        with patch.object(scraper_manager, "_head_check", return_value=500):
            scraper = await scraper_manager._select_auto_scraper(
                "https://error.example.com", config
            )
        from services.playwright_scraper import PlaywrightScraper

        assert isinstance(scraper, PlaywrightScraper)


class TestAutoFallback:
    """Tests for AUTO mode fallback behavior."""

    @pytest.mark.asyncio
    async def test_httpx_failure_falls_back_to_playwright(self, scraper_manager):
        """When HttpxScraper fails in AUTO mode, should fall back to Playwright."""
        failed_result = ScrapeResult(
            url="https://example.com",
            status=ScrapeStatus.FAILED,
            error="HTTP 403",
            scraper_used="HttpxScraper",
        )
        success_result = ScrapeResult(
            url="https://example.com",
            html_content="<html>content</html>",
            text_content="content",
            status=ScrapeStatus.COMPLETED,
            scraper_used="PlaywrightScraper",
        )

        from services.httpx_scraper import HttpxScraper
        from services.playwright_scraper import PlaywrightScraper

        mock_httpx = MagicMock(spec=HttpxScraper)
        mock_httpx.scrape = AsyncMock(return_value=failed_result)
        mock_httpx.name = "HttpxScraper"

        mock_playwright = MagicMock(spec=PlaywrightScraper)
        mock_playwright.scrape = AsyncMock(return_value=success_result)
        mock_playwright.name = "PlaywrightScraper"

        with (
            patch.object(
                scraper_manager,
                "_select_auto_scraper",
                return_value=mock_httpx,
            ),
            patch.object(
                scraper_manager,
                "_get_or_create_playwright",
                return_value=mock_playwright,
            ),
        ):
            result = await scraper_manager.scrape_url(
                "https://example.com", scraper_type=ScraperType.AUTO
            )

        assert result.status == ScrapeStatus.COMPLETED
        assert result.scraper_used == "PlaywrightScraper"
        # Verify fallback was called with is_fallback=True
        call_kwargs = mock_playwright.scrape.call_args
        assert call_kwargs[1].get("is_fallback") is True


class TestScraperManagerBrowserReady:
    """Tests for is_browser_ready."""

    def test_browser_ready_no_playwright(self, scraper_manager):
        """When no Playwright scraper exists, browser is 'ready' (nothing to wait for)."""
        assert scraper_manager._playwright_scraper is None
        assert scraper_manager.is_browser_ready() is True

    def test_browser_ready_initialized(self, scraper_manager):
        """When Playwright scraper is initialized, is_browser_ready returns True."""
        mock_pw = MagicMock()
        mock_pw.is_healthy.return_value = True
        scraper_manager._playwright_scraper = mock_pw
        assert scraper_manager.is_browser_ready() is True

    def test_browser_not_ready(self, scraper_manager):
        """When Playwright scraper exists but is not usable, returns False."""
        mock_pw = MagicMock()
        mock_pw.is_healthy.return_value = False
        scraper_manager._playwright_scraper = mock_pw
        assert scraper_manager.is_browser_ready() is False

    def test_a_disconnected_browser_is_not_ready(self):
        """`_initialized` only records that a launch once succeeded.

        A Chromium that has since died (OOM-kill is the realistic case on this
        single-replica pod) left it True, so readiness kept advertising a
        browser that was gone.
        """
        from services.playwright_scraper import PlaywrightScraper

        scraper = PlaywrightScraper()
        scraper._initialized = True
        scraper.browser = MagicMock()
        scraper.browser.is_connected.return_value = False
        assert scraper.is_healthy() is False

        scraper.browser.is_connected.return_value = True
        assert scraper.is_healthy() is True


class TestHeadCheck:
    """Tests for _head_check method."""

    @pytest.mark.asyncio
    async def test_head_check_success(self, scraper_manager, base_url):
        """HEAD check against local server returns status code."""
        status = await scraper_manager._head_check(f"{base_url}/simple")
        assert status == 200

    @pytest.mark.asyncio
    async def test_head_check_500(self, scraper_manager, base_url):
        """HEAD check against 500 page returns 500."""
        status = await scraper_manager._head_check(f"{base_url}/error500")
        assert status == 500

    @pytest.mark.asyncio
    async def test_head_check_failure(self, scraper_manager):
        """HEAD check against unreachable URL returns None."""
        status = await scraper_manager._head_check("http://127.0.0.1:1")
        assert status is None

    @pytest.mark.asyncio
    async def test_head_check_attaches_ssrf_hook(self, scraper_manager):
        """The pre-check client also carries the per-hop SSRF guard (it follows
        redirects too)."""
        import httpx

        captured: dict = {}
        real = httpx.AsyncClient

        def _capture(*a, **k):
            captured.update(k)
            return real(*a, **k)

        with patch("services.scraper_manager.httpx.AsyncClient", side_effect=_capture):
            await scraper_manager._head_check("http://127.0.0.1:1")
        assert captured.get("event_hooks", {}).get("request")


class TestAutoSelectionOtherStatus:
    """Tests for AUTO mode with other status codes (not 2xx, 403, 429, 5xx, None)."""

    @pytest.mark.asyncio
    async def test_auto_head_301_uses_httpx(self, scraper_manager):
        """AUTO with HEAD returning 301 (redirect) uses httpx."""
        config = ScrapeConfig()
        with patch.object(scraper_manager, "_head_check", return_value=301):
            scraper = await scraper_manager._select_auto_scraper(
                "https://redirect.example.com", config
            )
        from services.httpx_scraper import HttpxScraper

        assert isinstance(scraper, HttpxScraper)

    @pytest.mark.asyncio
    async def test_auto_head_404_uses_httpx(self, scraper_manager):
        """AUTO with HEAD returning 404 uses httpx (not in 403/429 blacklist)."""
        config = ScrapeConfig()
        with patch.object(scraper_manager, "_head_check", return_value=404):
            scraper = await scraper_manager._select_auto_scraper(
                "https://notfound.example.com", config
            )
        from services.httpx_scraper import HttpxScraper

        assert isinstance(scraper, HttpxScraper)


class TestGetScraperAutoFallback:
    """Tests for _get_scraper with AUTO type."""

    @pytest.mark.asyncio
    async def test_get_scraper_auto_falls_back_to_httpx(self, scraper_manager):
        """_get_scraper with AUTO type falls back to httpx."""
        scraper = await scraper_manager._get_scraper(ScraperType.AUTO)
        from services.httpx_scraper import HttpxScraper

        assert isinstance(scraper, HttpxScraper)


class TestScraperManagerDefaultConfig:
    """Tests for scrape_url with default config creation."""

    @pytest.mark.asyncio
    async def test_scrape_url_creates_default_config(self, scraper_manager, mock_scrape_result):
        """When config is None, scrape_url creates default config."""
        mock_httpx = MagicMock()
        mock_httpx.scrape = AsyncMock(return_value=mock_scrape_result)
        mock_httpx.name = "HttpxScraper"

        with patch.object(scraper_manager, "_get_scraper", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_httpx
            result = await scraper_manager.scrape_url(
                "https://example.com", scraper_type=ScraperType.HTTPX
            )
        assert result.status == ScrapeStatus.COMPLETED


class TestScraperManagerTimeoutPassthrough:
    """The resolved timeout must reach scraper.scrape as an explicit kwarg so the
    scraper reads a request-local value, not the shared (race-prone) self.config."""

    @pytest.mark.asyncio
    async def test_timeout_kwarg_passed_to_scraper(self, scraper_manager, mock_scrape_result):
        mock_scraper = MagicMock()
        mock_scraper.scrape = AsyncMock(return_value=mock_scrape_result)
        mock_scraper.name = "PlaywrightScraper"

        config = ScrapeConfig(timeout=42.0)
        with patch.object(scraper_manager, "_get_scraper", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_scraper
            await scraper_manager.scrape_url(
                "https://example.com", scraper_type=ScraperType.PLAYWRIGHT, config=config
            )

        assert mock_scraper.scrape.call_args.kwargs["timeout"] == 42.0

    @pytest.mark.asyncio
    async def test_timeout_override_kwarg_wins(self, scraper_manager, mock_scrape_result):
        """An explicit timeout kwarg overrides config.timeout end-to-end."""
        mock_scraper = MagicMock()
        mock_scraper.scrape = AsyncMock(return_value=mock_scrape_result)
        mock_scraper.name = "PlaywrightScraper"

        config = ScrapeConfig(timeout=42.0)
        with patch.object(scraper_manager, "_get_scraper", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_scraper
            await scraper_manager.scrape_url(
                "https://example.com",
                scraper_type=ScraperType.PLAYWRIGHT,
                config=config,
                timeout=7,
            )

        assert mock_scraper.scrape.call_args.kwargs["timeout"] == 7


class TestGetScraperManager:
    """Tests for get_scraper_manager singleton."""

    def test_get_scraper_manager_singleton(self):
        import services.scraper_manager as sm

        sm._scraper_manager = None

        manager1 = get_scraper_manager()
        manager2 = get_scraper_manager()

        assert manager1 is manager2
        assert isinstance(manager1, ScraperManager)

        # Cleanup
        sm._scraper_manager = None


class TestConcurrencyCap:
    @pytest.mark.asyncio
    async def test_semaphore_bounds_inflight_scrapes(self):
        """scrape_url respects the global concurrency semaphore (wires the
        previously-dead scraping_max_concurrent setting)."""
        mgr = ScraperManager()
        mgr._concurrency = asyncio.Semaphore(1)

        inflight = 0
        max_inflight = 0

        class SlowScraper:
            name = "slow"

            async def scrape(self, url, **kwargs):
                nonlocal inflight, max_inflight
                inflight += 1
                max_inflight = max(max_inflight, inflight)
                await asyncio.sleep(0.05)
                inflight -= 1
                return ScrapeResult(url=url, status=ScrapeStatus.COMPLETED, scraper_used="slow")

        with (
            patch.object(mgr, "_get_scraper", AsyncMock(return_value=SlowScraper())),
            patch("services.scraper_manager.assert_allowed_url", MagicMock()),
        ):
            await asyncio.gather(
                *[
                    mgr.scrape_url(f"https://example{i}.com", scraper_type=ScraperType.HTTPX)
                    for i in range(4)
                ]
            )

        assert max_inflight == 1

    def test_semaphore_sized_from_settings(self):
        mgr = ScraperManager()
        # Default scraping_max_concurrent is 16.
        assert mgr._concurrency._value == 16


_RICH_HTML = (
    "<html><head><title>T</title>"
    '<script type="application/ld+json">'
    '{"@context":"https://schema.org","@type":"Article","headline":"H",'
    '"author":{"@type":"Person","name":"Jane Doe"}}</script>'
    '<meta property="og:title" content="OG"/></head>'
    "<body><article><h1>Main Heading</h1>"
    "<p>This is a sufficiently long article body so the recall guard keeps it as "
    "real main content rather than discarding it as boilerplate noise.</p>"
    "</article></body></html>"
)


class TestEnrichExtraction:
    """scrape_url centrally derives markdown_content + structured_data."""

    @pytest.mark.asyncio
    async def test_completed_result_is_enriched(self, scraper_manager):
        rich = ScrapeResult(
            url="https://example.com/a",
            html_content=_RICH_HTML,
            text_content="Main Heading ...",
            status=ScrapeStatus.COMPLETED,
            scraper_used="HttpxScraper",
            metadata={"final_url": "https://example.com/a"},
        )
        scraper = MagicMock()
        scraper.scrape = AsyncMock(return_value=rich)
        scraper.name = "HttpxScraper"

        with (
            patch.object(scraper_manager, "_get_scraper", AsyncMock(return_value=scraper)),
            patch("services.scraper_manager.assert_allowed_url", MagicMock()),
        ):
            result = await scraper_manager.scrape_url(
                "https://example.com/a", scraper_type=ScraperType.HTTPX
            )

        assert result.markdown_content is not None
        assert "#" in result.markdown_content  # markdown heading marker
        assert "Main Heading" in result.markdown_content
        assert "json-ld" in result.structured_data
        assert result.structured_data["json-ld"][0]["author"]["name"] == "Jane Doe"

    @pytest.mark.asyncio
    async def test_failed_result_not_enriched(self, scraper_manager):
        failed = ScrapeResult(
            url="https://example.com/x",
            html_content=_RICH_HTML,  # html present but status FAILED → skip
            status=ScrapeStatus.FAILED,
            error="boom",
            scraper_used="HttpxScraper",
        )
        scraper = MagicMock()
        scraper.scrape = AsyncMock(return_value=failed)
        scraper.name = "HttpxScraper"

        with (
            patch.object(scraper_manager, "_get_scraper", AsyncMock(return_value=scraper)),
            patch("services.scraper_manager.assert_allowed_url", MagicMock()),
        ):
            result = await scraper_manager.scrape_url(
                "https://example.com/x", scraper_type=ScraperType.HTTPX
            )

        assert result.markdown_content is None
        assert result.structured_data == {}


class TestCurlCffiAutoSelection:
    """An explicit curl_cffi request resolves to the CurlCffiScraper instance.
    (AUTO 403/429 → curl_cffi is covered in TestAutoSelection.)"""

    @pytest.mark.asyncio
    async def test_explicit_curl_cffi_type(self, scraper_manager):
        from services.curl_cffi_scraper import CurlCffiScraper

        scraper = await scraper_manager._get_scraper(ScraperType.CURL_CFFI, ScrapeConfig())
        assert isinstance(scraper, CurlCffiScraper)
        assert scraper.name == "CurlCffiScraper"


class TestBrowserOnlyCapabilityRouting:
    """virtual_scroll and stealth only work in a browser.

    Routing them to httpx would ignore them silently and return a
    first-viewport (or unpatched) result reported as a full success.
    """

    @pytest.mark.asyncio
    async def test_auto_virtual_scroll_uses_playwright(self, scraper_manager):
        from schemas.scrape import VirtualScrollConfig
        from services.playwright_scraper import PlaywrightScraper

        config = ScrapeConfig(virtual_scroll=VirtualScrollConfig())
        scraper = await scraper_manager._select_auto_scraper("https://example.com", config)
        assert isinstance(scraper, PlaywrightScraper)

    @pytest.mark.asyncio
    async def test_auto_stealth_uses_playwright(self, scraper_manager):
        from services.playwright_scraper import PlaywrightScraper

        config = ScrapeConfig(stealth=True)
        scraper = await scraper_manager._select_auto_scraper("https://example.com", config)
        assert isinstance(scraper, PlaywrightScraper)


class TestStealthEscalation:
    """A curl_cffi failure is the strongest hostile-site signal AUTO produces."""

    async def _run_fallback(self, scraper_manager, failing_scraper):
        from services.base_scraper import ScrapeResult
        from services.base_scraper import ScrapeStatus as SvcStatus

        failed = ScrapeResult(url="https://x.com", status=SvcStatus.FAILED, scraper_used="f")
        ok = ScrapeResult(
            url="https://x.com",
            html_content="<html>ok</html>",
            status=SvcStatus.COMPLETED,
            scraper_used="PlaywrightScraper",
        )
        failing_scraper.scrape = AsyncMock(return_value=failed)
        fallback = MagicMock()
        fallback.name = "PlaywrightScraper"
        fallback.scrape = AsyncMock(return_value=ok)

        config = ScrapeConfig()
        with (
            patch.object(
                scraper_manager, "_select_auto_scraper", AsyncMock(return_value=failing_scraper)
            ),
            patch.object(scraper_manager, "_get_or_create_playwright", return_value=fallback),
            patch("services.scraper_manager.settings.ssrf_protection_enabled", False),
        ):
            await scraper_manager.scrape_url("https://x.com", config=config)
        return fallback.scrape.call_args.kwargs["config"], config

    @pytest.mark.asyncio
    async def test_curl_cffi_failure_escalates_to_stealth(self, scraper_manager):
        from services.curl_cffi_scraper import CurlCffiScraper

        scraper = MagicMock(spec=CurlCffiScraper)
        scraper.name = "CurlCffiScraper"
        used_config, _ = await self._run_fallback(scraper_manager, scraper)
        assert used_config.stealth is True

    @pytest.mark.asyncio
    async def test_plain_httpx_failure_does_not_escalate_to_stealth(self, scraper_manager):
        """A plain httpx failure is usually a JS-dependent page, not an anti-bot
        block — paying for stealth there is cost with no signal behind it."""
        from services.httpx_scraper import HttpxScraper

        scraper = MagicMock(spec=HttpxScraper)
        scraper.name = "HttpxScraper"
        used_config, _ = await self._run_fallback(scraper_manager, scraper)
        assert used_config.stealth is False

    @pytest.mark.asyncio
    async def test_escalation_does_not_mutate_the_callers_config(self, scraper_manager):
        """Mutating it would leak stealth into every later URL of a bulk scrape."""
        from services.curl_cffi_scraper import CurlCffiScraper

        scraper = MagicMock(spec=CurlCffiScraper)
        scraper.name = "CurlCffiScraper"
        used_config, original = await self._run_fallback(scraper_manager, scraper)
        assert used_config is not original
        assert original.stealth is False
