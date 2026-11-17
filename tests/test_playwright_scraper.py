"""Tests for PlaywrightScraper service."""

import asyncio
import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from playwright.async_api import TimeoutError as PlaywrightTimeoutError

from services.base_scraper import AuthConfig, ScrapeConfig, ScrapeStatus
from services.playwright_scraper import PlaywrightScraper


@pytest.fixture
def scrape_config():
    """Create basic scrape configuration."""
    return ScrapeConfig(
        timeout=30.0,
        screenshot=False,
        user_agent="TestAgent/1.0",
        headers={"X-Test": "test"},
        cookies={"session": "test123"},
    )


@pytest.fixture
def mock_playwright():
    """Mock Playwright components."""
    mock_pw = MagicMock()
    mock_browser = MagicMock()
    mock_page = MagicMock()
    mock_context = MagicMock()
    mock_response = MagicMock()

    # Setup async methods
    mock_page.goto = AsyncMock(return_value=mock_response)
    mock_page.content = AsyncMock(return_value="<html><body>Test</body></html>")
    mock_page.title = AsyncMock(return_value="Test Page")
    mock_page.evaluate = AsyncMock(return_value="Test text content")
    mock_page.set_extra_http_headers = AsyncMock()
    mock_page.wait_for_selector = AsyncMock()
    mock_page.screenshot = AsyncMock(return_value=b"fake_screenshot_bytes")
    mock_page.route = AsyncMock()
    mock_page.close = AsyncMock()

    # Setup page properties
    mock_page.url = "https://example.com"
    mock_page.context = mock_context
    mock_context.add_cookies = AsyncMock()

    # Setup response
    mock_response.status = 200

    # Setup browser
    mock_browser.new_page = AsyncMock(return_value=mock_page)
    mock_browser.close = AsyncMock()

    # Setup playwright
    mock_pw.chromium.launch = AsyncMock(return_value=mock_browser)
    mock_pw.stop = AsyncMock()

    return {
        "playwright": mock_pw,
        "browser": mock_browser,
        "page": mock_page,
        "context": mock_context,
        "response": mock_response,
    }


class TestPlaywrightScraperInitialization:
    """Test scraper initialization."""

    def test_init_default(self):
        """Test initialization with default config."""
        scraper = PlaywrightScraper()
        assert scraper.headless is True
        assert scraper.playwright is None
        assert scraper.browser is None
        assert scraper._initialized is False
        assert scraper.name == "PlaywrightScraper"

    def test_init_with_config(self, scrape_config):
        """Test initialization with custom config."""
        scraper = PlaywrightScraper(config=scrape_config, headless=False)
        assert scraper.headless is False
        assert scraper.config == scrape_config
        assert scraper.config.timeout == 30.0
        assert scraper.config.user_agent == "TestAgent/1.0"

    def test_is_initialized(self):
        """Test is_initialized method."""
        scraper = PlaywrightScraper()
        assert scraper.is_initialized() is False
        scraper._initialized = True
        assert scraper.is_initialized() is True


class TestPlaywrightScraperBrowserManagement:
    """Test browser lifecycle management."""

    @pytest.mark.asyncio
    async def test_ensure_browser_first_call(self, mock_playwright):
        """Test browser initialization on first call."""
        scraper = PlaywrightScraper()

        with patch("services.playwright_scraper.async_playwright") as mock_async_pw:
            mock_async_pw.return_value.start = AsyncMock(return_value=mock_playwright["playwright"])

            await scraper._ensure_browser()

            assert scraper._initialized is True
            assert scraper.playwright is not None
            assert scraper.browser is not None
            mock_playwright["playwright"].chromium.launch.assert_called_once()

    @pytest.mark.asyncio
    async def test_ensure_browser_already_initialized(self, mock_playwright):
        """Test that _ensure_browser skips initialization if already done."""
        scraper = PlaywrightScraper()
        scraper._initialized = True
        scraper.playwright = mock_playwright["playwright"]
        scraper.browser = mock_playwright["browser"]

        await scraper._ensure_browser()

        # Should not call launch again
        mock_playwright["playwright"].chromium.launch.assert_not_called()

    @pytest.mark.asyncio
    async def test_concurrent_ensure_browser_launches_once(self, mock_playwright):
        """Cold-start race: concurrent first-requests must launch exactly one
        browser (else the first is orphaned)."""
        scraper = PlaywrightScraper()

        async def slow_launch(*_a, **_k):
            await asyncio.sleep(0.05)  # widen the race window
            return mock_playwright["browser"]

        mock_playwright["playwright"].chromium.launch = AsyncMock(side_effect=slow_launch)

        with patch("services.playwright_scraper.async_playwright") as mock_async_pw:
            mock_async_pw.return_value.start = AsyncMock(return_value=mock_playwright["playwright"])
            await asyncio.gather(
                scraper._ensure_browser(),
                scraper._ensure_browser(),
                scraper._ensure_browser(),
            )

        mock_playwright["playwright"].chromium.launch.assert_called_once()
        assert scraper._initialized is True

    @pytest.mark.asyncio
    async def test_launch_failure_resets_state(self, mock_playwright):
        """A launch failure must not leak a half-started Playwright and must
        leave state clean so a later request can retry."""
        scraper = PlaywrightScraper()
        mock_playwright["playwright"].chromium.launch = AsyncMock(
            side_effect=RuntimeError("launch boom")
        )

        with patch("services.playwright_scraper.async_playwright") as mock_async_pw:
            mock_async_pw.return_value.start = AsyncMock(return_value=mock_playwright["playwright"])
            with pytest.raises(RuntimeError):
                await scraper._ensure_browser()

        assert scraper._initialized is False
        assert scraper.playwright is None
        assert scraper.browser is None
        mock_playwright["playwright"].stop.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_close_browser(self, mock_playwright):
        """Test browser cleanup."""
        scraper = PlaywrightScraper()
        scraper._initialized = True
        scraper.playwright = mock_playwright["playwright"]
        scraper.browser = mock_playwright["browser"]

        await scraper.close()

        mock_playwright["browser"].close.assert_called_once()
        mock_playwright["playwright"].stop.assert_called_once()
        assert scraper.browser is None
        assert scraper.playwright is None
        assert scraper._initialized is False

    @pytest.mark.asyncio
    async def test_close_not_initialized(self):
        """Test close when browser not initialized."""
        scraper = PlaywrightScraper()
        # Should not raise any errors
        await scraper.close()
        assert scraper._initialized is False


class TestPlaywrightScraperScraping:
    """Test main scraping functionality."""

    @pytest.mark.asyncio
    async def test_scrape_url_success(self, scrape_config, mock_playwright):
        """Test successful URL scraping."""
        scraper = PlaywrightScraper(config=scrape_config)

        # Text now comes from extract_main_text(page.content()), not page.evaluate.
        # evaluate is called for metadata, links, images (in that order).
        mock_playwright["page"].evaluate = AsyncMock(
            side_effect=[
                {"description": "Test description"},  # metadata
                [{"text": "Link 1", "url": "https://example.com/1", "rel": None}],  # Links
                [
                    {
                        "src": "https://example.com/img.jpg",
                        "alt": "Test",
                        "title": "",
                        "width": 100,
                        "height": 100,
                    }
                ],  # Images
            ]
        )

        with patch("services.playwright_scraper.async_playwright") as mock_async_pw:
            mock_async_pw.return_value.start = AsyncMock(return_value=mock_playwright["playwright"])

            result = await scraper.scrape("https://example.com")

            assert result.status == ScrapeStatus.COMPLETED
            assert result.url == "https://example.com"
            assert result.html_content == "<html><body>Test</body></html>"
            # html_content is tiny → trafilatura yields little → baseline returns "Test".
            assert result.text_content == "Test"
            assert result.title == "Test Page"
            assert result.metadata["status_code"] == 200
            assert result.metadata["final_url"] == "https://example.com"
            assert len(result.links) == 1
            assert result.links[0]["type"] == "internal"
            assert len(result.images) == 1
            assert result.scraper_used == "PlaywrightScraper"
            assert result.processing_time_seconds > 0
            assert result.error is None

            # Verify page was closed
            mock_playwright["page"].close.assert_called_once()

    @pytest.mark.asyncio
    async def test_scrape_absent_title_is_none(self, scrape_config, mock_playwright):
        """page.title() returns "" for a titleless page → result.title is None."""
        scraper = PlaywrightScraper(config=scrape_config)
        mock_playwright["page"].title = AsyncMock(return_value="")
        mock_playwright["page"].evaluate = AsyncMock(side_effect=[{}, [], []])

        with patch("services.playwright_scraper.async_playwright") as mock_async_pw:
            mock_async_pw.return_value.start = AsyncMock(return_value=mock_playwright["playwright"])

            result = await scraper.scrape("https://example.com")

            assert result.title is None

    @pytest.mark.asyncio
    async def test_scrape_url_with_custom_headers(self, mock_playwright):
        """Test scraping with custom headers."""
        config = ScrapeConfig(headers={"X-Custom": "value"})
        scraper = PlaywrightScraper(config=config)

        mock_playwright["page"].evaluate = AsyncMock(side_effect=[{}, [], []])

        with patch("services.playwright_scraper.async_playwright") as mock_async_pw:
            mock_async_pw.return_value.start = AsyncMock(return_value=mock_playwright["playwright"])

            await scraper.scrape("https://example.com")

            mock_playwright["page"].set_extra_http_headers.assert_called_once_with(
                {"X-Custom": "value"}
            )

    @pytest.mark.asyncio
    async def test_scrape_url_with_cookies(self, mock_playwright):
        """Test scraping with cookies."""
        config = ScrapeConfig(cookies={"session": "abc123", "user": "test"})
        scraper = PlaywrightScraper(config=config)

        mock_playwright["page"].evaluate = AsyncMock(side_effect=[{}, [], []])

        with patch("services.playwright_scraper.async_playwright") as mock_async_pw:
            mock_async_pw.return_value.start = AsyncMock(return_value=mock_playwright["playwright"])

            await scraper.scrape("https://example.com")

            # Verify cookies were added
            mock_playwright["context"].add_cookies.assert_called_once()
            cookies_arg = mock_playwright["context"].add_cookies.call_args[0][0]
            assert len(cookies_arg) == 2
            assert {
                "name": "session",
                "value": "abc123",
                "url": "https://example.com",
            } in cookies_arg
            assert {
                "name": "user",
                "value": "test",
                "url": "https://example.com",
            } in cookies_arg

    @pytest.mark.asyncio
    async def test_per_call_config_overrides_instance_config(self, mock_playwright):
        """A ``config`` kwarg is request-local and wins over the shared self.config.

        The instance was built with NO cookies/auth; the per-request config carries
        both. Reading from the passed config (not self.config) is what stops a
        concurrent request from leaking its auth/cookies through the shared
        singleton.
        """
        # Shared instance state has nothing request-specific.
        scraper = PlaywrightScraper(config=ScrapeConfig())

        request_cfg = ScrapeConfig(
            cookies={"sid": "req-local"},
            auth_config=AuthConfig(type="bearer", token="req-token"),
        )

        mock_playwright["page"].evaluate = AsyncMock(side_effect=[{}, [], []])

        with patch("services.playwright_scraper.async_playwright") as mock_async_pw:
            mock_async_pw.return_value.start = AsyncMock(return_value=mock_playwright["playwright"])

            await scraper.scrape("https://example.com", config=request_cfg)

            cookies_arg = mock_playwright["context"].add_cookies.call_args[0][0]
            assert {
                "name": "sid",
                "value": "req-local",
                "url": "https://example.com",
            } in cookies_arg

            headers_arg = mock_playwright["page"].set_extra_http_headers.call_args[0][0]
            assert headers_arg.get("Authorization") == "Bearer req-token"

    @pytest.mark.asyncio
    async def test_scrape_url_with_invalid_cookies(self, mock_playwright):
        """Test scraping with invalid cookies type."""
        config = ScrapeConfig(cookies="invalid")  # Not a dict
        scraper = PlaywrightScraper(config=config)

        mock_playwright["page"].evaluate = AsyncMock(side_effect=[{}, [], []])

        with patch("services.playwright_scraper.async_playwright") as mock_async_pw:
            mock_async_pw.return_value.start = AsyncMock(return_value=mock_playwright["playwright"])

            await scraper.scrape("https://example.com")

            # Should not add cookies
            mock_playwright["context"].add_cookies.assert_not_called()

    @pytest.mark.asyncio
    async def test_scrape_url_with_screenshot(self, mock_playwright):
        """Test scraping with screenshot capture."""
        config = ScrapeConfig(screenshot=True)
        scraper = PlaywrightScraper(config=config)

        mock_playwright["page"].evaluate = AsyncMock(side_effect=[{}, [], []])

        with patch("services.playwright_scraper.async_playwright") as mock_async_pw:
            mock_async_pw.return_value.start = AsyncMock(return_value=mock_playwright["playwright"])

            result = await scraper.scrape("https://example.com")

            assert result.screenshot_base64 is not None
            # Verify it's valid base64
            decoded = base64.b64decode(result.screenshot_base64)
            assert decoded == b"fake_screenshot_bytes"
            mock_playwright["page"].screenshot.assert_called_once_with(full_page=True)

    @pytest.mark.asyncio
    async def test_scrape_url_with_viewport_settings(self, mock_playwright):
        """Test scraping with custom viewport settings."""
        scraper = PlaywrightScraper()

        mock_playwright["page"].evaluate = AsyncMock(side_effect=[{}, [], []])

        with patch("services.playwright_scraper.async_playwright") as mock_async_pw:
            mock_async_pw.return_value.start = AsyncMock(return_value=mock_playwright["playwright"])

            await scraper.scrape("https://example.com", viewport_width=1280, viewport_height=720)

            # Verify viewport was set
            call_kwargs = mock_playwright["browser"].new_page.call_args[1]
            assert call_kwargs["viewport"] == {"width": 1280, "height": 720}

    @pytest.mark.asyncio
    async def test_scrape_url_with_wait_for_selector(self, mock_playwright):
        """Test scraping with wait_for_selector option."""
        scraper = PlaywrightScraper()

        mock_playwright["page"].evaluate = AsyncMock(side_effect=[{}, [], []])

        with patch("services.playwright_scraper.async_playwright") as mock_async_pw:
            mock_async_pw.return_value.start = AsyncMock(return_value=mock_playwright["playwright"])

            await scraper.scrape("https://example.com", wait_for_selector=".content")

            mock_playwright["page"].wait_for_selector.assert_called_once_with(
                ".content",
                timeout=30000,  # Default timeout
            )


class TestPlaywrightScraperSsrfRoute:
    """The page.route SSRF guard must cover EVERY request, not just navigations."""

    async def _capture_route_handler(self, mock_playwright):
        """Run one scrape to register the guard, return the handler page.route got."""
        scraper = PlaywrightScraper()
        mock_playwright["page"].evaluate = AsyncMock(side_effect=[{}, [], []])
        with patch("services.playwright_scraper.async_playwright") as mock_async_pw:
            mock_async_pw.return_value.start = AsyncMock(return_value=mock_playwright["playwright"])
            await scraper.scrape("https://example.com")
        route_calls = mock_playwright["page"].route.call_args_list
        assert route_calls, "page.route was never called — SSRF guard not registered"
        pattern, handler = route_calls[0].args
        assert pattern == "**/*"
        return handler

    @pytest.mark.asyncio
    async def test_non_navigation_request_to_internal_host_is_aborted(self, mock_playwright):
        """A subresource/fetch/XHR (is_navigation_request()==False) to a disallowed
        internal host must be ABORTED — the old code only guarded navigations, so a
        scraped page could fetch() the cloud-metadata endpoint (blind SSRF)."""
        handler = await self._capture_route_handler(mock_playwright)

        route = MagicMock()
        route.request = MagicMock()
        route.request.url = "http://169.254.169.254/latest/meta-data/"
        # Prove the OLD narrowing (guard only when navigation) would have let this by.
        route.request.is_navigation_request = MagicMock(return_value=False)
        route.abort = AsyncMock()
        route.continue_ = AsyncMock()

        await handler(route)

        route.abort.assert_awaited_once_with("blockedbyclient")
        route.continue_.assert_not_called()

    @pytest.mark.asyncio
    async def test_non_navigation_request_to_allowed_host_continues(self, mock_playwright):
        """A subresource to a public/allowed host is still continued — the guard
        must not break legitimate asset loading. Uses a public IP literal so the
        assertion is DNS-free and deterministic in a sandboxed runner."""
        handler = await self._capture_route_handler(mock_playwright)

        route = MagicMock()
        route.request = MagicMock()
        route.request.url = "https://8.8.8.8/app.js"
        route.request.is_navigation_request = MagicMock(return_value=False)
        route.abort = AsyncMock()
        route.continue_ = AsyncMock()

        await handler(route)

        route.continue_.assert_awaited_once()
        route.abort.assert_not_called()


class TestPlaywrightScraperErrorHandling:
    """Test error handling scenarios."""

    @pytest.mark.asyncio
    async def test_scrape_timeout_error(self, mock_playwright):
        """Test handling of Playwright timeout errors."""
        scraper = PlaywrightScraper()

        mock_playwright["page"].goto = AsyncMock(
            side_effect=PlaywrightTimeoutError("Navigation timeout")
        )

        with patch("services.playwright_scraper.async_playwright") as mock_async_pw:
            mock_async_pw.return_value.start = AsyncMock(return_value=mock_playwright["playwright"])

            result = await scraper.scrape("https://example.com")

            assert result.status == ScrapeStatus.FAILED
            assert "Timeout" in result.error
            assert result.url == "https://example.com"
            assert result.scraper_used == "PlaywrightScraper"
            # Page should still be closed
            mock_playwright["page"].close.assert_called_once()

    @pytest.mark.asyncio
    async def test_scrape_connection_error(self, mock_playwright):
        """Test handling of connection errors."""
        scraper = PlaywrightScraper()

        mock_playwright["page"].goto = AsyncMock(side_effect=Exception("Connection refused"))

        with patch("services.playwright_scraper.async_playwright") as mock_async_pw:
            mock_async_pw.return_value.start = AsyncMock(return_value=mock_playwright["playwright"])

            result = await scraper.scrape("https://example.com")

            assert result.status == ScrapeStatus.FAILED
            assert "Connection refused" in result.error
            assert result.scraper_used == "PlaywrightScraper"

    @pytest.mark.asyncio
    async def test_scrape_browser_initialization_error(self):
        """Test error during browser initialization."""
        scraper = PlaywrightScraper()

        with patch("services.playwright_scraper.async_playwright") as mock_async_pw:
            mock_async_pw.return_value.start = AsyncMock(
                side_effect=Exception("Browser launch failed")
            )

            result = await scraper.scrape("https://example.com")

            assert result.status == ScrapeStatus.FAILED
            assert "Browser launch failed" in result.error

    @pytest.mark.asyncio
    async def test_scrape_screenshot_error(self, mock_playwright):
        """Test handling of screenshot capture errors."""
        config = ScrapeConfig(screenshot=True)
        scraper = PlaywrightScraper(config=config)

        # Screenshot fails
        mock_playwright["page"].screenshot = AsyncMock(side_effect=Exception("Screenshot failed"))
        mock_playwright["page"].evaluate = AsyncMock(side_effect=[{}, [], []])

        with patch("services.playwright_scraper.async_playwright") as mock_async_pw:
            mock_async_pw.return_value.start = AsyncMock(return_value=mock_playwright["playwright"])

            result = await scraper.scrape("https://example.com")

            # Should still succeed but with no screenshot
            assert result.status == ScrapeStatus.COMPLETED
            assert result.screenshot_base64 is None


class TestPlaywrightScraperContentExtraction:
    """Test content extraction methods."""

    @pytest.mark.asyncio
    async def test_extract_metadata_success(self, mock_playwright):
        """Test successful metadata extraction."""
        scraper = PlaywrightScraper()

        metadata = {"og:title": "Test", "description": "Test page"}
        mock_playwright["page"].evaluate = AsyncMock(return_value=metadata)

        result = await scraper._extract_metadata(mock_playwright["page"])

        assert result == metadata

    @pytest.mark.asyncio
    async def test_extract_metadata_error(self, mock_playwright):
        """Test metadata extraction error handling."""
        scraper = PlaywrightScraper()

        mock_playwright["page"].evaluate = AsyncMock(side_effect=Exception("Evaluation failed"))

        result = await scraper._extract_metadata(mock_playwright["page"])

        assert result == {}

    @pytest.mark.asyncio
    async def test_extract_links_success(self, mock_playwright):
        """Test successful link extraction."""
        scraper = PlaywrightScraper()

        links = [
            {"text": "Internal", "url": "https://example.com/page", "rel": None},
            {"text": "External", "url": "https://other.com/page", "rel": "nofollow"},
        ]
        mock_playwright["page"].evaluate = AsyncMock(return_value=links)

        result = await scraper._extract_links(mock_playwright["page"], "https://example.com")

        assert len(result) == 2
        assert result[0]["type"] == "internal"
        assert result[1]["type"] == "external"

    @pytest.mark.asyncio
    async def test_extract_links_error(self, mock_playwright):
        """Test link extraction error handling."""
        scraper = PlaywrightScraper()

        mock_playwright["page"].evaluate = AsyncMock(side_effect=Exception("Failed"))

        result = await scraper._extract_links(mock_playwright["page"], "https://example.com")

        assert result == []

    @pytest.mark.asyncio
    async def test_extract_images_success(self, mock_playwright):
        """Test successful image extraction."""
        scraper = PlaywrightScraper()

        images = [
            {
                "src": "https://example.com/img.jpg",
                "alt": "Test",
                "title": "Title",
                "width": 100,
                "height": 100,
            }
        ]
        mock_playwright["page"].evaluate = AsyncMock(return_value=images)

        result = await scraper._extract_images(mock_playwright["page"])

        assert len(result) == 1
        assert result[0]["src"] == "https://example.com/img.jpg"

    @pytest.mark.asyncio
    async def test_extract_images_error(self, mock_playwright):
        """Test image extraction error handling."""
        scraper = PlaywrightScraper()

        mock_playwright["page"].evaluate = AsyncMock(side_effect=Exception("Failed"))

        result = await scraper._extract_images(mock_playwright["page"])

        assert result == []

    @pytest.mark.asyncio
    async def test_capture_screenshot_base64_success(self, mock_playwright):
        """Test successful screenshot capture and base64 encoding."""
        scraper = PlaywrightScraper()

        screenshot_bytes = b"screenshot_data"
        mock_playwright["page"].screenshot = AsyncMock(return_value=screenshot_bytes)

        result = await scraper._capture_screenshot_base64(mock_playwright["page"])

        expected = base64.b64encode(screenshot_bytes).decode("utf-8")
        assert result == expected

    @pytest.mark.asyncio
    async def test_capture_screenshot_base64_error(self, mock_playwright):
        """Test screenshot capture error handling."""
        scraper = PlaywrightScraper()

        mock_playwright["page"].screenshot = AsyncMock(side_effect=Exception("Screenshot failed"))

        result = await scraper._capture_screenshot_base64(mock_playwright["page"])

        assert result is None


class TestPlaywrightScraperResourceCleanup:
    """Test resource cleanup scenarios."""

    @pytest.mark.asyncio
    async def test_page_closed_on_success(self, mock_playwright):
        """Test that page is closed after successful scrape."""
        scraper = PlaywrightScraper()

        mock_playwright["page"].evaluate = AsyncMock(side_effect=[{}, [], []])

        with patch("services.playwright_scraper.async_playwright") as mock_async_pw:
            mock_async_pw.return_value.start = AsyncMock(return_value=mock_playwright["playwright"])

            await scraper.scrape("https://example.com")

            mock_playwright["page"].close.assert_called_once()

    @pytest.mark.asyncio
    async def test_page_closed_on_error(self, mock_playwright):
        """Test that page is closed even when scraping fails."""
        scraper = PlaywrightScraper()

        mock_playwright["page"].goto = AsyncMock(side_effect=Exception("Failed"))

        with patch("services.playwright_scraper.async_playwright") as mock_async_pw:
            mock_async_pw.return_value.start = AsyncMock(return_value=mock_playwright["playwright"])

            await scraper.scrape("https://example.com")

            # Page should still be closed in finally block
            mock_playwright["page"].close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_cleans_all_resources(self, mock_playwright):
        """Test that close properly cleans up all resources."""
        scraper = PlaywrightScraper()
        scraper._initialized = True
        scraper.playwright = mock_playwright["playwright"]
        scraper.browser = mock_playwright["browser"]

        await scraper.close()

        mock_playwright["browser"].close.assert_called_once()
        mock_playwright["playwright"].stop.assert_called_once()
        assert scraper.browser is None
        assert scraper.playwright is None
        assert scraper._initialized is False


class TestPlaywrightScraperDownloadImages:
    """Tests for PlaywrightScraper with download_images option."""

    @pytest.mark.asyncio
    async def test_scrape_with_download_images(self, mock_playwright):
        """download_images=True triggers enrich_images_with_content for Playwright."""
        config = ScrapeConfig(download_images=True)
        scraper = PlaywrightScraper(config=config)

        mock_playwright["page"].evaluate = AsyncMock(
            side_effect=[
                {"description": "Test"},  # metadata
                [],  # links
                [
                    {
                        "src": "https://example.com/img.png",
                        "alt": "Test",
                        "title": "",
                        "width": 100,
                        "height": 50,
                    }
                ],  # images
            ]
        )

        enriched = [
            {
                "src": "https://example.com/img.png",
                "alt": "Test",
                "title": "",
                "width": 100,
                "height": 50,
                "content_base64": "iVBORbase64data",
                "mime_type": "image/png",
                "byte_size": 1024,
            }
        ]

        with (
            patch("services.playwright_scraper.async_playwright") as mock_async_pw,
            patch(
                "services.playwright_scraper.enrich_images_with_content",
                return_value=enriched,
            ) as mock_enrich,
        ):
            mock_async_pw.return_value.start = AsyncMock(return_value=mock_playwright["playwright"])
            result = await scraper.scrape("https://example.com", download_images=True)

        mock_enrich.assert_awaited_once()
        assert result.status == ScrapeStatus.COMPLETED
        assert len(result.images) == 1
        assert result.images[0]["content_base64"] == "iVBORbase64data"
        assert result.images[0]["mime_type"] == "image/png"
        assert result.images[0]["byte_size"] == 1024

    @pytest.mark.asyncio
    async def test_scrape_without_download_images(self, mock_playwright):
        """download_images=False does not trigger enrichment for Playwright."""
        config = ScrapeConfig(download_images=False)
        scraper = PlaywrightScraper(config=config)

        mock_playwright["page"].evaluate = AsyncMock(
            side_effect=[
                {},  # metadata
                [],  # links
                [{"src": "https://example.com/img.png", "alt": "Test", "title": ""}],  # images
            ]
        )

        with (
            patch("services.playwright_scraper.async_playwright") as mock_async_pw,
            patch(
                "services.playwright_scraper.enrich_images_with_content",
            ) as mock_enrich,
        ):
            mock_async_pw.return_value.start = AsyncMock(return_value=mock_playwright["playwright"])
            result = await scraper.scrape("https://example.com")

        mock_enrich.assert_not_awaited()
        assert result.status == ScrapeStatus.COMPLETED
        assert len(result.images) == 1
        assert "content_base64" not in result.images[0]


class TestPlaywrightScraperFallbackBehavior:
    """Tests for PlaywrightScraper fallback mode behavior."""

    @pytest.mark.asyncio
    async def test_fallback_mode_uses_domcontentloaded(self, mock_playwright):
        """When is_fallback=True, wait_until is forced to domcontentloaded."""
        scraper = PlaywrightScraper()

        mock_playwright["page"].evaluate = AsyncMock(side_effect=[{}, [], []])

        with patch("services.playwright_scraper.async_playwright") as mock_async_pw:
            mock_async_pw.return_value.start = AsyncMock(return_value=mock_playwright["playwright"])
            await scraper.scrape("https://example.com", is_fallback=True)

        # Verify goto was called with domcontentloaded
        goto_call = mock_playwright["page"].goto.call_args
        assert goto_call.kwargs["wait_until"] == "domcontentloaded"

    @pytest.mark.asyncio
    async def test_non_fallback_uses_load(self, mock_playwright):
        """When is_fallback=False (default), wait_until defaults to 'load'.

        Regression: 'networkidle' was the old default and never settles on live
        SPAs, blocking goto for the whole timeout. 'load' is the safe bounded
        default; networkidle remains available as an explicit opt-in.
        """
        scraper = PlaywrightScraper()

        mock_playwright["page"].evaluate = AsyncMock(side_effect=[{}, [], []])

        with patch("services.playwright_scraper.async_playwright") as mock_async_pw:
            mock_async_pw.return_value.start = AsyncMock(return_value=mock_playwright["playwright"])
            await scraper.scrape("https://example.com")

        goto_call = mock_playwright["page"].goto.call_args
        assert goto_call.kwargs["wait_until"] == "load"

    @pytest.mark.asyncio
    async def test_explicit_wait_until_is_honored(self, mock_playwright):
        """An explicit wait_until (incl. networkidle) overrides the 'load' default."""
        scraper = PlaywrightScraper()

        mock_playwright["page"].evaluate = AsyncMock(side_effect=[{}, [], []])

        with patch("services.playwright_scraper.async_playwright") as mock_async_pw:
            mock_async_pw.return_value.start = AsyncMock(return_value=mock_playwright["playwright"])
            await scraper.scrape("https://example.com", wait_until="networkidle")

        goto_call = mock_playwright["page"].goto.call_args
        assert goto_call.kwargs["wait_until"] == "networkidle"

    @pytest.mark.asyncio
    async def test_null_response_status(self, mock_playwright):
        """When page.goto returns None, status_code is set to None in metadata."""
        scraper = PlaywrightScraper()

        mock_playwright["page"].goto = AsyncMock(return_value=None)
        mock_playwright["page"].evaluate = AsyncMock(side_effect=[{}, [], []])

        with patch("services.playwright_scraper.async_playwright") as mock_async_pw:
            mock_async_pw.return_value.start = AsyncMock(return_value=mock_playwright["playwright"])
            result = await scraper.scrape("https://example.com")

        assert result.status == ScrapeStatus.COMPLETED
        assert result.metadata["status_code"] is None
        assert result.metadata["final_url"] == "https://example.com"

    @pytest.mark.asyncio
    async def test_no_headers_no_cookies(self, mock_playwright):
        """Scraper with empty headers/cookies does not call set_extra_http_headers or add_cookies."""
        config = ScrapeConfig(headers={}, cookies={})
        scraper = PlaywrightScraper(config=config)

        mock_playwright["page"].evaluate = AsyncMock(side_effect=[{}, [], []])

        with patch("services.playwright_scraper.async_playwright") as mock_async_pw:
            mock_async_pw.return_value.start = AsyncMock(return_value=mock_playwright["playwright"])
            result = await scraper.scrape("https://example.com")

        assert result.status == ScrapeStatus.COMPLETED
        mock_playwright["page"].set_extra_http_headers.assert_not_called()
        mock_playwright["context"].add_cookies.assert_not_called()


class TestPlaywrightScraperHangHardening:
    """Regression tests for the unbounded-wait fixes (the live-SPA 'infinite hang').

    page.set_default_timeout bounds content()/title()/wait_*; the overall
    asyncio.wait_for budget is the backstop for page.evaluate, which has no
    timeout parameter and ignores the page default.
    """

    @pytest.mark.asyncio
    async def test_overall_budget_aborts_hung_evaluate(self, mock_playwright):
        """A page.evaluate that never returns is aborted by the overall budget and
        yields a FAILED result instead of hanging the request forever."""
        scraper = PlaywrightScraper(config=ScrapeConfig(timeout=0.2))

        async def _hang(*args, **kwargs):
            await asyncio.sleep(3600)

        # Real async function (not AsyncMock) so the await genuinely blocks.
        mock_playwright["page"].evaluate = _hang

        with patch("services.playwright_scraper.async_playwright") as mock_async_pw:
            mock_async_pw.return_value.start = AsyncMock(return_value=mock_playwright["playwright"])
            result = await scraper.scrape("https://example.com", timeout=0.2, extraction_margin=0.3)

        assert result.status == ScrapeStatus.FAILED
        assert "budget exceeded" in result.error
        # Cleanup must still run despite the cancellation.
        mock_playwright["page"].close.assert_called_once()

    @pytest.mark.asyncio
    async def test_page_default_timeouts_are_set(self, mock_playwright):
        """set_default_timeout / set_default_navigation_timeout are set to the
        effective timeout in ms so content()/title()/wait_* are bounded."""
        scraper = PlaywrightScraper(config=ScrapeConfig(timeout=12.0))

        mock_playwright["page"].evaluate = AsyncMock(side_effect=[{}, [], []])

        with patch("services.playwright_scraper.async_playwright") as mock_async_pw:
            mock_async_pw.return_value.start = AsyncMock(return_value=mock_playwright["playwright"])
            await scraper.scrape("https://example.com")

        mock_playwright["page"].set_default_timeout.assert_called_once_with(12000)
        mock_playwright["page"].set_default_navigation_timeout.assert_called_once_with(12000)

    @pytest.mark.asyncio
    async def test_explicit_timeout_kwarg_overrides_config(self, mock_playwright):
        """The per-call timeout kwarg (passed by ScraperManager) wins over
        self.config.timeout for the navigation timeout — request-local, race-safe."""
        scraper = PlaywrightScraper(config=ScrapeConfig(timeout=99.0))

        mock_playwright["page"].evaluate = AsyncMock(side_effect=[{}, [], []])

        with patch("services.playwright_scraper.async_playwright") as mock_async_pw:
            mock_async_pw.return_value.start = AsyncMock(return_value=mock_playwright["playwright"])
            await scraper.scrape("https://example.com", timeout=5)

        goto_call = mock_playwright["page"].goto.call_args
        assert goto_call.kwargs["timeout"] == 5000
        mock_playwright["page"].set_default_timeout.assert_called_once_with(5000)

    @pytest.mark.asyncio
    async def test_safe_close_swallows_close_error(self, mock_playwright):
        """A page whose close() raises does not blow up the scrape; the original
        failure result is still returned."""
        scraper = PlaywrightScraper()
        mock_playwright["page"].goto = AsyncMock(side_effect=Exception("nav boom"))
        mock_playwright["page"].close = AsyncMock(side_effect=Exception("close boom"))

        with patch("services.playwright_scraper.async_playwright") as mock_async_pw:
            mock_async_pw.return_value.start = AsyncMock(return_value=mock_playwright["playwright"])
            result = await scraper.scrape("https://example.com")

        assert result.status == ScrapeStatus.FAILED
        assert "nav boom" in result.error


class TestPlaywrightSsrfRouteGuard:
    """The page.route navigation guard that aborts redirects to blocked hosts."""

    async def _handler_after_scrape(self, mock_playwright):
        scraper = PlaywrightScraper()
        mock_playwright["page"].evaluate = AsyncMock(side_effect=[{}, [], []])
        with patch("services.playwright_scraper.async_playwright") as mock_async_pw:
            mock_async_pw.return_value.start = AsyncMock(return_value=mock_playwright["playwright"])
            await scraper.scrape("https://example.com")
        # The route handler is the 2nd positional arg of page.route("**/*", handler)
        mock_playwright["page"].route.assert_awaited()
        return mock_playwright["page"].route.call_args[0][1]

    def _route(self, *, url, navigation):
        route = MagicMock()
        route.abort = AsyncMock()
        route.continue_ = AsyncMock()
        req = MagicMock()
        req.is_navigation_request = MagicMock(return_value=navigation)
        req.url = url
        route.request = req
        return route

    @pytest.mark.asyncio
    async def test_navigation_to_blocked_host_aborted(self, mock_playwright):
        handler = await self._handler_after_scrape(mock_playwright)
        route = self._route(url="http://169.254.169.254/latest/meta-data/", navigation=True)
        await handler(route)
        route.abort.assert_awaited_once()
        route.continue_.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_navigation_to_public_host_continues(self, mock_playwright):
        handler = await self._handler_after_scrape(mock_playwright)
        route = self._route(url="https://93.184.216.34/page", navigation=True)
        await handler(route)
        route.continue_.assert_awaited_once()
        route.abort.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_subresource_to_internal_host_is_blocked(self, mock_playwright):
        # SSRF: in-page subresource/fetch requests are now guarded too (previously
        # only navigation was), so a scraped page cannot fetch() an internal host.
        handler = await self._handler_after_scrape(mock_playwright)
        route = self._route(url="http://10.0.0.1/asset.png", navigation=False)
        await handler(route)
        route.abort.assert_awaited_once()
        route.continue_.assert_not_awaited()
