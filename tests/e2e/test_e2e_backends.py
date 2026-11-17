"""
E2E tests for scraper backends, auth middleware, and manager logic.

These tests exercise code paths that the basic httpx-only E2E tests miss:
- PlaywrightScraper (mocked browser — Chromium not installed in CI)
- ScrapyScraper (via API with mocked _run_crawl_in_executor)
- ScraperManager auto-selection, fallback chains
- Auth middleware (token validation, trusted networks, rejection)
- Main app lifespan, CORS config, OpenAPI schema
- Base scraper utility methods
"""

import asyncio
import base64
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

# ``e2e``, not ``integration``: in-process full stack (ASGI app + a local HTTP
# server, Playwright mocked), no external network and no live service, so it runs
# in the default suite and CI. ``integration`` is reserved for tests that need a
# live deployment or a real browser.
pytestmark = pytest.mark.e2e

# The crawl runs in a spawned ProcessPoolExecutor child, so ``_run_crawl`` is
# pickled across the process boundary — a mock patched onto it cannot be
# (PicklingError). Patch the awaitable seam in the parent instead, as the
# module documents (``_run_crawl_in_executor``).
_CRAWL_SEAM = "services.scrapy_scraper._run_crawl_in_executor"


async def _drain(agen) -> list:
    """Collect an async generator (e.g. ``WebSpider.start()``) into a list."""
    return [item async for item in agen]


# ─── Playwright Scraper E2E ─────────────────────────────────────────────────


class TestPlaywrightScrapeE2E:
    """POST /api/scrape/url with scraper_type=playwright — mocked browser."""

    @pytest.fixture
    def _mock_playwright(self):
        """Mock Playwright so we don't need a real browser binary."""
        mock_page = AsyncMock()
        mock_page.goto = AsyncMock(return_value=MagicMock(status=200))
        mock_page.content = AsyncMock(
            return_value="<html><head><title>PW Page</title></head>"
            "<body><h1>Hello PW</h1><p>rendered</p></body></html>"
        )
        mock_page.title = AsyncMock(return_value="PW Page")
        mock_page.url = "http://example.com/test"
        # page.evaluate call order of the CURRENT scraper: metadata, links,
        # images. text_content is no longer an innerText evaluate — since #32
        # (8da9b19) it is extract_main_text(page.content()), so a leading
        # innerText entry here shifts every result by one (metadata became a str).
        mock_page.evaluate = AsyncMock(
            side_effect=[
                # 1st call: metadata extraction
                {"description": "pw test", "og:title": "PW OG"},
                # 2nd call: link extraction
                [
                    {"text": "Link1", "url": "http://example.com/a", "rel": None},
                    {"text": "Ext", "url": "http://other.com/b", "rel": None},
                ],
                # 3rd call: image extraction
                [
                    {
                        "src": "http://example.com/img.png",
                        "alt": "Logo",
                        "title": "The Logo",
                        "width": 100,
                        "height": 50,
                    }
                ],
            ]
        )
        mock_page.screenshot = AsyncMock(return_value=b"\x89PNG fake screenshot bytes")
        mock_page.close = AsyncMock()
        mock_page.wait_for_selector = AsyncMock()
        mock_page.context = MagicMock()
        mock_page.context.add_cookies = AsyncMock()

        mock_browser = AsyncMock()
        mock_browser.new_page = AsyncMock(return_value=mock_page)
        mock_browser.close = AsyncMock()

        mock_pw_instance = AsyncMock()
        mock_pw_instance.chromium = MagicMock()
        mock_pw_instance.chromium.launch = AsyncMock(return_value=mock_browser)
        mock_pw_instance.stop = AsyncMock()

        mock_async_pw = AsyncMock()
        mock_async_pw.__aenter__ = AsyncMock(return_value=mock_pw_instance)
        mock_async_pw.__aexit__ = AsyncMock()
        mock_async_pw.start = AsyncMock(return_value=mock_pw_instance)

        with patch("services.playwright_scraper.async_playwright", return_value=mock_async_pw):
            yield {
                "page": mock_page,
                "browser": mock_browser,
                "pw_instance": mock_pw_instance,
            }

    @pytest.mark.asyncio
    async def test_playwright_scrape_basic(self, async_client, _mock_playwright):
        """Full Playwright scrape flow with mocked browser."""
        resp = await async_client.post(
            "/api/scrape/url",
            json={
                "url": "http://example.com/test",
                "scraper_type": "playwright",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        assert data["scraper_used"] == "PlaywrightScraper"
        assert data["title"] == "PW Page"
        assert "Hello PW" in data["text_content"]
        assert data["processing_time_seconds"] > 0

    @pytest.mark.asyncio
    async def test_playwright_scrape_with_screenshot(self, async_client, _mock_playwright):
        """Playwright scrape captures screenshot when requested."""
        resp = await async_client.post(
            "/api/scrape/url",
            json={
                "url": "http://example.com/test",
                "scraper_type": "playwright",
                "screenshot": True,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        assert data["screenshot_base64"] is not None
        assert len(data["screenshot_base64"]) > 0

    @pytest.mark.asyncio
    async def test_playwright_scrape_with_cookies(self, async_client, _mock_playwright):
        """Cookies are set on the browser context."""
        resp = await async_client.post(
            "/api/scrape/url",
            json={
                "url": "http://example.com/test",
                "scraper_type": "playwright",
                "cookies": {"session": "abc123"},
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        # Verify cookies were added
        _mock_playwright["page"].context.add_cookies.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_playwright_scrape_with_custom_headers(self, async_client, _mock_playwright):
        """Custom headers are set on the page."""
        resp = await async_client.post(
            "/api/scrape/url",
            json={
                "url": "http://example.com/test",
                "scraper_type": "playwright",
                "headers": {"X-Custom": "value"},
            },
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "completed"
        _mock_playwright["page"].set_extra_http_headers.assert_awaited()

    @pytest.mark.asyncio
    async def test_playwright_scrape_with_viewport(self, async_client, _mock_playwright):
        """Custom viewport dimensions are passed to browser."""
        resp = await async_client.post(
            "/api/scrape/url",
            json={
                "url": "http://example.com/test",
                "scraper_type": "playwright",
                "viewport_width": 800,
                "viewport_height": 600,
            },
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "completed"
        call_kwargs = _mock_playwright["browser"].new_page.call_args
        assert call_kwargs.kwargs["viewport"] == {"width": 800, "height": 600}

    @pytest.mark.asyncio
    async def test_playwright_scrape_extracts_links(self, async_client, _mock_playwright):
        """Links are extracted and classified as internal/external."""
        resp = await async_client.post(
            "/api/scrape/url",
            json={
                "url": "http://example.com/test",
                "scraper_type": "playwright",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        links = data["links"]
        assert len(links) == 2
        internal = [lnk for lnk in links if lnk["type"] == "internal"]
        external = [lnk for lnk in links if lnk["type"] == "external"]
        assert len(internal) == 1
        assert len(external) == 1
        assert internal[0]["text"] == "Link1"

    @pytest.mark.asyncio
    async def test_playwright_scrape_extracts_images(self, async_client, _mock_playwright):
        """Images are extracted with metadata."""
        resp = await async_client.post(
            "/api/scrape/url",
            json={
                "url": "http://example.com/test",
                "scraper_type": "playwright",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        images = data["images"]
        assert len(images) == 1
        assert images[0]["alt"] == "Logo"
        assert images[0]["title"] == "The Logo"
        assert images[0]["width"] == 100
        assert images[0]["height"] == 50

    @pytest.mark.asyncio
    async def test_playwright_scrape_extracts_metadata(self, async_client, _mock_playwright):
        """Metadata including OG tags is extracted."""
        resp = await async_client.post(
            "/api/scrape/url",
            json={
                "url": "http://example.com/test",
                "scraper_type": "playwright",
            },
        )
        assert resp.status_code == 200
        meta = resp.json()["metadata"]
        assert meta["description"] == "pw test"
        assert meta["og:title"] == "PW OG"
        assert meta["status_code"] == 200
        assert meta["final_url"] == "http://example.com/test"

    @pytest.mark.asyncio
    async def test_playwright_timeout_returns_failed(self, async_client):
        """Playwright timeout error returns failed status, not 500."""
        from playwright.async_api import TimeoutError as PWTimeout

        mock_page = AsyncMock()
        mock_page.goto = AsyncMock(side_effect=PWTimeout("Navigation timeout 30000ms exceeded"))
        mock_page.close = AsyncMock()
        mock_page.context = MagicMock()

        mock_browser = AsyncMock()
        mock_browser.new_page = AsyncMock(return_value=mock_page)
        mock_browser.close = AsyncMock()

        mock_pw = AsyncMock()
        mock_pw.chromium = MagicMock()
        mock_pw.chromium.launch = AsyncMock(return_value=mock_browser)
        mock_pw.stop = AsyncMock()

        mock_async_pw = AsyncMock()
        mock_async_pw.start = AsyncMock(return_value=mock_pw)

        with patch("services.playwright_scraper.async_playwright", return_value=mock_async_pw):
            resp = await async_client.post(
                "/api/scrape/url",
                json={
                    "url": "http://example.com/slow",
                    "scraper_type": "playwright",
                    "timeout": 1.0,
                },
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "failed"
        assert "timeout" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_playwright_general_error_returns_failed(self, async_client):
        """General Playwright error returns failed status."""
        mock_pw = AsyncMock()
        mock_pw.chromium = MagicMock()
        mock_pw.chromium.launch = AsyncMock(side_effect=RuntimeError("Browser launch failed"))
        mock_pw.stop = AsyncMock()

        mock_async_pw = AsyncMock()
        mock_async_pw.start = AsyncMock(return_value=mock_pw)

        with patch("services.playwright_scraper.async_playwright", return_value=mock_async_pw):
            resp = await async_client.post(
                "/api/scrape/url",
                json={
                    "url": "http://example.com/test",
                    "scraper_type": "playwright",
                },
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "failed"
        assert "Browser launch failed" in data["error"]

    @pytest.mark.asyncio
    async def test_playwright_scrape_with_wait_for_selector(self, async_client, _mock_playwright):
        """wait_for_selector is passed to page."""
        resp = await async_client.post(
            "/api/scrape/url",
            json={
                "url": "http://example.com/test",
                "scraper_type": "playwright",
                "wait_for_selector": "#content",
            },
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "completed"
        _mock_playwright["page"].wait_for_selector.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_playwright_close(self, async_client, _mock_playwright):
        """PlaywrightScraper.close() cleans up browser resources."""
        from services.playwright_scraper import PlaywrightScraper

        scraper = PlaywrightScraper()
        scraper.browser = _mock_playwright["browser"]
        scraper.playwright = _mock_playwright["pw_instance"]
        scraper._initialized = True

        await scraper.close()
        assert scraper._initialized is False
        assert scraper.browser is None
        assert scraper.playwright is None


# ─── Scrapy Scraper E2E ─────────────────────────────────────────────────────


class TestScrapyScrapeE2E:
    """POST /api/scrape/url with scraper_type=scrapy — mocked _run_crawl_in_executor."""

    @pytest.mark.asyncio
    async def test_scrapy_scrape_basic(self, async_client, base_url):
        """Scrapy scrape returns content via mocked crawl."""
        items = [
            {
                "url": f"{base_url}/simple",
                "html_content": "<html><body><h1>Scrapy Result</h1></body></html>",
                "text_content": "Scrapy Result",
                "title": "Scrapy Title",
                "metadata": {"status_code": 200, "final_url": f"{base_url}/simple"},
                "links": [
                    {"text": "About", "url": f"{base_url}/about", "rel": "", "type": "internal"}
                ],
                "images": [{"src": f"{base_url}/img.png", "alt": "Pic", "title": ""}],
                "depth": 0,
            }
        ]
        with patch(_CRAWL_SEAM, new=AsyncMock(return_value=(items, []))):
            resp = await async_client.post(
                "/api/scrape/url",
                json={
                    "url": f"{base_url}/simple",
                    "scraper_type": "scrapy",
                },
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        assert data["scraper_used"] == "ScrapyScraper"
        assert data["title"] == "Scrapy Title"
        assert len(data["links"]) == 1
        assert len(data["images"]) == 1

    @pytest.mark.asyncio
    async def test_scrapy_scrape_no_items_returns_failed(self, async_client, base_url):
        """Scrapy returning no items → failed status."""
        with patch(
            _CRAWL_SEAM,
            new=AsyncMock(return_value=([], ["Connection refused"])),
        ):
            resp = await async_client.post(
                "/api/scrape/url",
                json={
                    "url": f"{base_url}/simple",
                    "scraper_type": "scrapy",
                },
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "failed"
        assert "Connection refused" in data["error"]

    @pytest.mark.asyncio
    async def test_scrapy_scrape_exception_returns_failed(self, async_client, base_url):
        """Scrapy exception → failed status."""
        with patch(
            _CRAWL_SEAM,
            new=AsyncMock(side_effect=RuntimeError("Twisted reactor error")),
        ):
            resp = await async_client.post(
                "/api/scrape/url",
                json={
                    "url": f"{base_url}/simple",
                    "scraper_type": "scrapy",
                },
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "failed"
        assert "RuntimeError" in data["error"]

    @pytest.mark.asyncio
    async def test_scrapy_scrape_multiple_items(self, async_client, base_url):
        """Scrapy with multiple crawled pages uses the first item."""
        items = [
            {
                "url": f"{base_url}/simple",
                "html_content": "<html><body>Page 1</body></html>",
                "text_content": "Page 1",
                "title": "First Page",
                "metadata": {"status_code": 200, "final_url": f"{base_url}/simple"},
                "links": [],
                "images": [],
                "depth": 0,
            },
            {
                "url": f"{base_url}/page2",
                "html_content": "<html><body>Page 2</body></html>",
                "text_content": "Page 2",
                "title": "Second Page",
                "metadata": {"status_code": 200, "final_url": f"{base_url}/page2"},
                "links": [],
                "images": [],
                "depth": 1,
            },
        ]
        with patch(_CRAWL_SEAM, new=AsyncMock(return_value=(items, []))):
            resp = await async_client.post(
                "/api/scrape/url",
                json={
                    "url": f"{base_url}/simple",
                    "scraper_type": "scrapy",
                },
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["title"] == "First Page"

    @pytest.mark.asyncio
    async def test_scrapy_close(self):
        """ScrapyScraper.close() is a no-op that doesn't error."""
        from services.scrapy_scraper import ScrapyScraper

        scraper = ScrapyScraper()
        await scraper.close()  # Should not raise


# ─── Scraper Manager E2E ────────────────────────────────────────────────────


class TestScraperManagerE2E:
    """ScraperManager auto-selection, fallback chains, lifecycle."""

    @pytest.mark.asyncio
    async def test_auto_selects_httpx_on_200_head(self, async_client, base_url):
        """AUTO mode selects httpx when HEAD returns 200."""
        resp = await async_client.post(
            "/api/scrape/url",
            json={"url": f"{base_url}/simple", "scraper_type": "auto"},
        )
        assert resp.status_code == 200
        assert resp.json()["scraper_used"] == "HttpxScraper"

    @pytest.mark.asyncio
    async def test_auto_selects_curl_cffi_on_403(self, async_client, base_url):
        """AUTO mode tries curl_cffi (Chrome impersonation) first when HEAD returns 403.

        Contract since #46 (8a8dadd, scraper_manager.py:182): a 403/429 means the
        site blocks plain clients, so curl_cffi is cheaper than a browser and runs
        first. The local server answers 200, so curl_cffi completes.
        """
        with patch(
            "services.scraper_manager.ScraperManager._head_check",
            return_value=403,
        ):
            resp = await async_client.post(
                "/api/scrape/url",
                json={"url": f"{base_url}/simple", "scraper_type": "auto"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        assert data["scraper_used"] == "CurlCffiScraper"

    @pytest.mark.asyncio
    async def test_auto_escalates_to_stealth_playwright_when_curl_cffi_fails_on_429(
        self, async_client, base_url
    ):
        """AUTO + HEAD 429 → curl_cffi; when it FAILs, escalate to Playwright with
        stealth forced on (#46, scraper_manager.py:298-318)."""
        from services.base_scraper import ScrapeResult, ScrapeStatus

        pw_scrape = AsyncMock(
            return_value=ScrapeResult(
                url=f"{base_url}/simple",
                html_content="<html><body>ok</body></html>",
                text_content="ok",
                title="OK",
                metadata={"status_code": 200, "final_url": f"{base_url}/simple"},
                status=ScrapeStatus.COMPLETED,
                scraper_used="PlaywrightScraper",
                processing_time_seconds=0.1,
            )
        )
        with (
            patch(
                "services.scraper_manager.ScraperManager._head_check",
                return_value=429,
            ),
            patch(
                "services.curl_cffi_scraper.CurlCffiScraper.scrape",
                new=AsyncMock(
                    return_value=ScrapeResult(
                        url=f"{base_url}/simple",
                        status=ScrapeStatus.FAILED,
                        error="HTTP 429",
                        scraper_used="CurlCffiScraper",
                    )
                ),
            ),
            patch("services.playwright_scraper.PlaywrightScraper.scrape", new=pw_scrape),
        ):
            resp = await async_client.post(
                "/api/scrape/url",
                json={"url": f"{base_url}/simple", "scraper_type": "auto"},
            )
        assert resp.status_code == 200
        assert resp.json()["scraper_used"] == "PlaywrightScraper"
        pw_scrape.assert_awaited_once()
        assert pw_scrape.await_args.kwargs["config"].stealth is True
        assert pw_scrape.await_args.kwargs["is_fallback"] is True

    @pytest.mark.asyncio
    async def test_auto_selects_playwright_on_head_failure(self, async_client, base_url):
        """AUTO mode selects Playwright when HEAD returns None (timeout)."""
        from services.base_scraper import ScrapeResult, ScrapeStatus

        with (
            patch(
                "services.scraper_manager.ScraperManager._head_check",
                return_value=None,
            ),
            patch(
                "services.playwright_scraper.PlaywrightScraper.scrape",
                return_value=ScrapeResult(
                    url=f"{base_url}/simple",
                    status=ScrapeStatus.COMPLETED,
                    scraper_used="PlaywrightScraper",
                    processing_time_seconds=0.1,
                    metadata={"status_code": 200, "final_url": f"{base_url}/simple"},
                ),
            ),
        ):
            resp = await async_client.post(
                "/api/scrape/url",
                json={"url": f"{base_url}/simple", "scraper_type": "auto"},
            )
        assert resp.status_code == 200
        assert resp.json()["scraper_used"] == "PlaywrightScraper"

    @pytest.mark.asyncio
    async def test_auto_selects_playwright_on_500(self, async_client, base_url):
        """AUTO mode selects Playwright when HEAD returns 500."""
        from services.base_scraper import ScrapeResult, ScrapeStatus

        with (
            patch(
                "services.scraper_manager.ScraperManager._head_check",
                return_value=500,
            ),
            patch(
                "services.playwright_scraper.PlaywrightScraper.scrape",
                return_value=ScrapeResult(
                    url=f"{base_url}/simple",
                    status=ScrapeStatus.COMPLETED,
                    scraper_used="PlaywrightScraper",
                    processing_time_seconds=0.1,
                    metadata={"status_code": 200, "final_url": f"{base_url}/simple"},
                ),
            ),
        ):
            resp = await async_client.post(
                "/api/scrape/url",
                json={"url": f"{base_url}/simple", "scraper_type": "auto"},
            )
        assert resp.status_code == 200
        assert resp.json()["scraper_used"] == "PlaywrightScraper"

    @pytest.mark.asyncio
    async def test_auto_selects_scrapy_for_follow_links(self, async_client, base_url):
        """AUTO mode selects Scrapy when follow_links is set."""
        items = [
            {
                "url": f"{base_url}/simple",
                "html_content": "<html><body>Crawled</body></html>",
                "text_content": "Crawled",
                "title": "Crawled Page",
                "metadata": {"status_code": 200, "final_url": f"{base_url}/simple"},
                "links": [],
                "images": [],
                "depth": 0,
            }
        ]
        with patch(_CRAWL_SEAM, new=AsyncMock(return_value=(items, []))):
            resp = await async_client.post(
                "/api/scrape/bulk",
                json={
                    "urls": [f"{base_url}/simple"],
                    "scraper_type": "auto",
                    "follow_links": True,
                    "max_depth": 2,
                },
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["results"][0]["scraper_used"] == "ScrapyScraper"

    @pytest.mark.asyncio
    async def test_auto_fallback_httpx_to_playwright(self, async_client, base_url):
        """AUTO mode falls back from httpx to Playwright on failure."""
        from services.base_scraper import ScrapeResult, ScrapeStatus

        with (
            patch(
                "services.httpx_scraper.HttpxScraper.scrape",
                return_value=ScrapeResult(
                    url=f"{base_url}/simple",
                    status=ScrapeStatus.FAILED,
                    error="HTTP 403: Forbidden",
                    scraper_used="HttpxScraper",
                    processing_time_seconds=0.1,
                ),
            ),
            patch(
                "services.playwright_scraper.PlaywrightScraper.scrape",
                return_value=ScrapeResult(
                    url=f"{base_url}/simple",
                    html_content="<html><body>Fallback worked</body></html>",
                    text_content="Fallback worked",
                    title="Fallback Page",
                    metadata={"status_code": 200, "final_url": f"{base_url}/simple"},
                    status=ScrapeStatus.COMPLETED,
                    scraper_used="PlaywrightScraper",
                    processing_time_seconds=0.5,
                ),
            ),
        ):
            resp = await async_client.post(
                "/api/scrape/url",
                json={"url": f"{base_url}/simple", "scraper_type": "auto"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        assert data["scraper_used"] == "PlaywrightScraper"
        assert data["title"] == "Fallback Page"

    @pytest.mark.asyncio
    async def test_manager_is_browser_ready_no_playwright(self):
        """is_browser_ready() returns True when no Playwright instance exists."""
        from services.scraper_manager import ScraperManager

        mgr = ScraperManager()
        assert mgr.is_browser_ready() is True

    @pytest.mark.asyncio
    async def test_manager_is_browser_ready_with_playwright(self):
        """is_browser_ready() delegates to PlaywrightScraper.is_healthy() — initialized
        AND still connected (contract since #50, 2111671; scraper_manager.py:93)."""
        from services.scraper_manager import ScraperManager

        mgr = ScraperManager()
        mock_pw = MagicMock()
        mock_pw.is_healthy.return_value = False
        mgr._playwright_scraper = mock_pw
        assert mgr.is_browser_ready() is False

        mock_pw.is_healthy.return_value = True
        assert mgr.is_browser_ready() is True

    @pytest.mark.asyncio
    async def test_manager_close_all_scrapers(self):
        """close() closes all scraper instances and nullifies them."""
        from services.scraper_manager import ScraperManager

        mgr = ScraperManager()
        mock_pw = AsyncMock()
        mock_httpx = AsyncMock()
        mock_scrapy = AsyncMock()
        mgr._playwright_scraper = mock_pw
        mgr._httpx_scraper = mock_httpx
        mgr._scrapy_scraper = mock_scrapy

        await mgr.close()

        mock_pw.close.assert_awaited_once()
        mock_httpx.close.assert_awaited_once()
        mock_scrapy.close.assert_awaited_once()
        assert mgr._playwright_scraper is None
        assert mgr._httpx_scraper is None
        assert mgr._scrapy_scraper is None

    @pytest.mark.asyncio
    async def test_manager_head_check_returns_status(self):
        """_head_check returns HTTP status code."""
        from services.scraper_manager import ScraperManager

        mgr = ScraperManager()

        mock_resp = MagicMock()
        mock_resp.status_code = 200

        with patch("services.scraper_manager.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.head = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock()
            mock_client_cls.return_value = mock_client

            result = await mgr._head_check("http://example.com")

        assert result == 200

    @pytest.mark.asyncio
    async def test_manager_head_check_returns_none_on_error(self):
        """_head_check returns None on exception."""
        from services.scraper_manager import ScraperManager

        mgr = ScraperManager()

        with patch("services.scraper_manager.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.head = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock()
            mock_client_cls.return_value = mock_client

            result = await mgr._head_check("http://example.com")

        assert result is None

    @pytest.mark.asyncio
    async def test_manager_get_or_create_reuses_instances(self):
        """Scraper instances are reused across calls."""
        from services.base_scraper import ScrapeConfig
        from services.scraper_manager import ScraperManager

        mgr = ScraperManager()
        config = ScrapeConfig()

        httpx1 = mgr._get_or_create_httpx(config)
        httpx2 = mgr._get_or_create_httpx(config)
        assert httpx1 is httpx2

        scrapy1 = mgr._get_or_create_scrapy(config)
        scrapy2 = mgr._get_or_create_scrapy(config)
        assert scrapy1 is scrapy2

        pw1 = mgr._get_or_create_playwright(config)
        pw2 = mgr._get_or_create_playwright(config)
        assert pw1 is pw2

    @pytest.mark.asyncio
    async def test_manager_singleton(self):
        """get_scraper_manager() returns the same instance."""
        import services.scraper_manager as sm

        old = sm._scraper_manager
        sm._scraper_manager = None
        try:
            mgr1 = sm.get_scraper_manager()
            mgr2 = sm.get_scraper_manager()
            assert mgr1 is mgr2
        finally:
            if sm._scraper_manager:
                await sm._scraper_manager.close()
            sm._scraper_manager = old


# ─── Auth Middleware E2E ─────────────────────────────────────────────────────


class TestAuthMiddlewareE2E:
    """Authentication middleware — token validation, trusted networks."""

    @pytest.mark.asyncio
    async def test_public_paths_skip_auth(self, async_client):
        """Public paths don't require auth."""
        for path in ["/", "/health", "/health/live", "/health/ready", "/docs", "/redoc"]:
            resp = await async_client.get(path)
            assert resp.status_code != 401, f"Path {path} should be public but got 401"

    @pytest.mark.asyncio
    async def test_protected_path_trusted_network(self, async_client, base_url):
        """Trusted network (127.0.0.1 via ASGI) bypasses auth."""
        # ASGI transport sets client to 127.0.0.1 which is in trusted CIDRs
        resp = await async_client.post(
            "/api/scrape/url",
            json={"url": f"{base_url}/simple", "scraper_type": "httpx"},
        )
        assert resp.status_code == 200  # Not 401

    @pytest.mark.asyncio
    async def test_protected_path_with_valid_bearer_token(self, async_client, base_url):
        """Valid Bearer token in X-Tenant-Id header authenticates."""
        with patch(
            "auth.middleware._introspect_license",
            return_value={"active": True, "tenant_id": "test-tenant", "subject_id": "user-1"},
        ):
            resp = await async_client.post(
                "/api/scrape/url",
                json={"url": f"{base_url}/simple", "scraper_type": "httpx"},
                headers={"X-Tenant-Id": "Bearer valid-token-123"},
            )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_protected_path_with_invalid_token(self, async_client, base_url):
        """Invalid token returns 401."""
        with patch(
            "auth.middleware._introspect_license",
            return_value={"active": False},
        ):
            # Need to simulate non-trusted network to trigger token validation path
            with patch("auth.middleware._is_trusted_network", return_value=False):
                resp = await async_client.post(
                    "/api/scrape/url",
                    json={"url": f"{base_url}/simple", "scraper_type": "httpx"},
                    headers={"X-Tenant-Id": "Bearer invalid-token"},
                )
        assert resp.status_code == 401
        data = resp.json()
        assert data["error"]["code"] == "AUTH_REQUIRED"
        assert "Invalid or expired" in data["error"]["message"]

    @pytest.mark.asyncio
    async def test_protected_path_token_missing_tenant(self, async_client, base_url):
        """Token valid but missing tenant context returns 401."""
        with patch(
            "auth.middleware._introspect_license",
            return_value={"active": True},  # No tenant_id or subject_id
        ):
            with patch("auth.middleware._is_trusted_network", return_value=False):
                resp = await async_client.post(
                    "/api/scrape/url",
                    json={"url": f"{base_url}/simple", "scraper_type": "httpx"},
                    headers={"X-Tenant-Id": "Bearer token-no-tenant"},
                )
        assert resp.status_code == 401
        assert "tenant context" in resp.json()["error"]["message"].lower()

    @pytest.mark.asyncio
    async def test_protected_path_no_token_untrusted_network(self, async_client, base_url):
        """No token + untrusted network → 401."""
        with patch("auth.middleware._is_trusted_network", return_value=False):
            resp = await async_client.post(
                "/api/scrape/url",
                json={"url": f"{base_url}/simple", "scraper_type": "httpx"},
            )
        assert resp.status_code == 401
        assert "Missing authentication token" in resp.json()["error"]["message"]

    @pytest.mark.asyncio
    async def test_protected_path_introspection_failure(self, async_client, base_url):
        """Introspection returning None → 401."""
        with patch(
            "auth.middleware._introspect_license",
            return_value=None,
        ):
            with patch("auth.middleware._is_trusted_network", return_value=False):
                resp = await async_client.post(
                    "/api/scrape/url",
                    json={"url": f"{base_url}/simple", "scraper_type": "httpx"},
                    headers={"X-Tenant-Id": "Bearer some-token"},
                )
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_trusted_network_with_custom_tenant(self, async_client, base_url):
        """Trusted network with X-Tenant-Id (non-Bearer) uses it as tenant."""
        resp = await async_client.post(
            "/api/scrape/url",
            json={"url": f"{base_url}/simple", "scraper_type": "httpx"},
            headers={"X-Tenant-Id": "my-custom-tenant"},
        )
        # ASGI transport is 127.0.0.1 = trusted, and X-Tenant-Id without "Bearer " prefix
        # is used as the tenant ID directly
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_auth_middleware_bypass_mode(self):
        """bypass_auth=True allows all requests."""
        from auth.middleware import AuthMiddleware
        from main import app

        # Check that app has the middleware configured
        # The existing app uses bypass_auth=False by default, test the class directly
        middleware = AuthMiddleware(app, protected_prefixes=["/api"], bypass_auth=True)
        assert middleware.bypass_auth is True


# ─── Auth Middleware Internal Functions ──────────────────────────────────────


class TestAuthMiddlewareInternals:
    """Direct tests on auth module functions."""

    def test_is_trusted_network_ipv4(self):
        """IPv4 addresses in RFC 1918 ranges are trusted."""
        from auth.middleware import _is_trusted_network

        assert _is_trusted_network("127.0.0.1") is True
        assert _is_trusted_network("192.168.1.100") is True
        assert _is_trusted_network("10.0.0.1") is True
        assert _is_trusted_network("172.16.0.1") is True

    def test_is_trusted_network_untrusted(self):
        """Public IPs are not trusted."""
        from auth.middleware import _is_trusted_network

        assert _is_trusted_network("8.8.8.8") is False
        assert _is_trusted_network("203.0.113.1") is False

    def test_is_trusted_network_none(self):
        """None host is not trusted."""
        from auth.middleware import _is_trusted_network

        assert _is_trusted_network(None) is False

    def test_is_trusted_network_invalid(self):
        """Invalid IP string is not trusted."""
        from auth.middleware import _is_trusted_network

        assert _is_trusted_network("not-an-ip") is False
        assert _is_trusted_network("") is False

    @pytest.mark.asyncio
    async def test_introspect_license_success(self):
        """Successful license introspection caches and returns data."""
        from auth.middleware import _introspect_license, _license_cache

        _license_cache.clear()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"active": True, "tenant_id": "t1"}

        with patch("auth.middleware.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock()
            mock_cls.return_value = mock_client

            result = await _introspect_license("test-token-123")

        assert result is not None
        assert result["active"] is True
        assert result["tenant_id"] == "t1"

    @pytest.mark.asyncio
    async def test_introspect_license_inactive(self):
        """Inactive license is returned but NOT cached."""
        from auth.middleware import _introspect_license, _license_cache

        _license_cache.clear()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"active": False}

        with patch("auth.middleware.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock()
            mock_cls.return_value = mock_client

            result = await _introspect_license("inactive-token")

        assert result == {"active": False}
        # Not cached since active=False
        assert len(_license_cache) == 0

    @pytest.mark.asyncio
    async def test_introspect_license_error(self):
        """Network error is an availability failure → raise (→ 503), not 401."""
        from auth.middleware import (
            LicenseServiceUnavailableError,
            _introspect_license,
            _license_cache,
        )

        _license_cache.clear()

        with patch("auth.middleware.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            # Like httpx.AsyncClient.__aexit__, return None (do NOT suppress). A bare
            # AsyncMock() returns a truthy MagicMock, which swallows the ConnectError.
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_cls.return_value = mock_client

            with pytest.raises(LicenseServiceUnavailableError):
                await _introspect_license("error-token")

    @pytest.mark.asyncio
    async def test_introspect_license_cache_hit(self):
        """Cached result is returned without HTTP call."""
        import hashlib

        from auth.middleware import _introspect_license, _license_cache

        _license_cache.clear()

        token = "cached-token"
        cache_key = f"lic:{hashlib.sha256(token.encode()).hexdigest()[:16]}"
        cached_data = {"active": True, "tenant_id": "cached-tenant"}
        # Positive cache stores (data, expires_at); seed a future expiry.
        import time

        _license_cache[cache_key] = (cached_data, time.time() + 300)

        # Should not make any HTTP call
        result = await _introspect_license(token)
        assert result == cached_data

        _license_cache.clear()


# ─── Main App E2E ───────────────────────────────────────────────────────────


class TestMainAppE2E:
    """Main app configuration, lifespan, OpenAPI."""

    @pytest.mark.asyncio
    async def test_root_endpoint(self, async_client):
        """Root returns service info."""
        resp = await async_client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["service"] == "webscraper-service"
        assert data["status"] == "online"
        assert "endpoints" in data
        assert data["endpoints"]["health"] == "/health"

    @pytest.mark.asyncio
    async def test_openapi_schema(self, async_client):
        """OpenAPI schema is accessible and has custom security scheme."""
        resp = await async_client.get("/openapi.json")
        assert resp.status_code == 200
        schema = resp.json()
        assert "components" in schema
        assert "securitySchemes" in schema["components"]
        assert "BearerAuth" in schema["components"]["securitySchemes"]
        assert "parameters" in schema["components"]
        assert "TenantIdHeader" in schema["components"]["parameters"]

    @pytest.mark.asyncio
    async def test_docs_endpoint(self, async_client):
        """Swagger docs are accessible."""
        resp = await async_client.get("/docs")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_redoc_endpoint(self, async_client):
        """ReDoc is accessible."""
        resp = await async_client.get("/redoc")
        assert resp.status_code == 200

    def test_cors_config_defaults(self):
        """CORS config uses defaults when env var not set."""
        from main import get_cors_config

        with patch.dict("os.environ", {}, clear=False):
            # Remove CORS_ALLOW_ORIGINS if set
            with patch("main.os.getenv", return_value=None):
                origins, creds = get_cors_config()
        assert creds is True
        assert "http://localhost:3000" in origins

    def test_cors_config_custom_origins(self):
        """CORS config parses custom origins."""
        from main import get_cors_config

        with patch("main.os.getenv", return_value="http://a.com,http://b.com"):
            origins, creds = get_cors_config()
        assert creds is True
        assert origins == ["http://a.com", "http://b.com"]

    def test_cors_config_wildcard(self):
        """CORS config with * disables credentials."""
        from main import get_cors_config

        with patch("main.os.getenv", return_value="*"):
            origins, creds = get_cors_config()
        assert creds is False
        assert origins == ["*"]


# ─── Base Scraper Utilities ──────────────────────────────────────────────────


class TestBaseScraperUtilities:
    """BaseScraper utility methods — URL handling."""

    @pytest.fixture
    def scraper(self):
        """Create a concrete subclass for testing."""
        from services.httpx_scraper import HttpxScraper

        return HttpxScraper()

    def test_is_valid_url_http(self, scraper):
        assert scraper.is_valid_url("http://example.com") is True

    def test_is_valid_url_https(self, scraper):
        assert scraper.is_valid_url("https://example.com/path") is True

    def test_is_valid_url_no_scheme(self, scraper):
        assert scraper.is_valid_url("example.com") is False

    def test_is_valid_url_no_netloc(self, scraper):
        assert scraper.is_valid_url("http://") is False

    def test_is_valid_url_empty(self, scraper):
        assert scraper.is_valid_url("") is False

    def test_is_valid_url_garbage(self, scraper):
        assert scraper.is_valid_url("not a url at all") is False

    def test_normalize_url_strips_fragment(self, scraper):
        assert scraper.normalize_url("http://example.com/page#section") == "http://example.com/page"

    def test_normalize_url_resolves_relative(self, scraper):
        result = scraper.normalize_url("/about", base_url="http://example.com")
        assert result == "http://example.com/about"

    def test_normalize_url_absolute_unchanged(self, scraper):
        result = scraper.normalize_url("http://other.com/page", base_url="http://example.com")
        assert result == "http://other.com/page"

    def test_normalize_url_no_base(self, scraper):
        result = scraper.normalize_url("/relative/path")
        assert result == "/relative/path"

    def test_should_scrape_url_valid(self, scraper):
        assert scraper.should_scrape_url("http://example.com/page") is True

    def test_should_scrape_url_invalid(self, scraper):
        assert scraper.should_scrape_url("not-a-url") is False

    def test_should_scrape_url_allowed_domain(self, scraper):
        assert (
            scraper.should_scrape_url("http://example.com/page", allowed_domains=["example.com"])
            is True
        )

    def test_should_scrape_url_blocked_domain(self, scraper):
        assert (
            scraper.should_scrape_url("http://evil.com/page", allowed_domains=["example.com"])
            is False
        )

    def test_should_scrape_url_subdomain_allowed(self, scraper):
        assert (
            scraper.should_scrape_url(
                "http://sub.example.com/page", allowed_domains=["example.com"]
            )
            is True
        )

    def test_extract_domain(self, scraper):
        assert scraper.extract_domain("http://example.com/path") == "example.com"
        assert scraper.extract_domain("https://sub.example.com:8080/path") == "sub.example.com:8080"

    def test_extract_domain_invalid(self, scraper):
        assert scraper.extract_domain("") == ""


# ─── ScrapeConfig Dataclass ─────────────────────────────────────────────────


class TestScrapeConfig:
    """ScrapeConfig defaults and post_init."""

    def test_defaults(self):
        from services.base_scraper import ScrapeConfig

        cfg = ScrapeConfig()
        # delay is opt-in: default changed 1.0 -> 0.0 in e0ab24a.
        assert cfg.delay == 0.0
        assert cfg.timeout == 30.0
        assert cfg.max_retries == 3
        assert cfg.follow_links is False
        assert cfg.max_depth == 1
        assert cfg.headers == {}
        assert cfg.cookies == {}
        assert cfg.screenshot is False
        assert cfg.download_images is False

    def test_post_init_initializes_none_dicts(self):
        from services.base_scraper import ScrapeConfig

        cfg = ScrapeConfig(headers=None, cookies=None)
        assert cfg.headers == {}
        assert cfg.cookies == {}

    def test_custom_values(self):
        from services.base_scraper import ScrapeConfig

        cfg = ScrapeConfig(
            delay=2.0,
            timeout=60.0,
            follow_links=True,
            max_depth=3,
            headers={"Accept": "text/html"},
            cookies={"session": "abc"},
        )
        assert cfg.delay == 2.0
        assert cfg.timeout == 60.0
        assert cfg.follow_links is True
        assert cfg.max_depth == 3
        assert cfg.headers == {"Accept": "text/html"}
        assert cfg.cookies == {"session": "abc"}


# ─── ScrapeResult Dataclass ─────────────────────────────────────────────────


class TestScrapeResult:
    """ScrapeResult defaults."""

    def test_defaults(self):
        from services.base_scraper import ScrapeResult, ScrapeStatus

        r = ScrapeResult(url="http://example.com")
        assert r.url == "http://example.com"
        assert r.html_content is None
        assert r.text_content is None
        assert r.title is None
        assert r.metadata == {}
        assert r.links == []
        assert r.images == []
        assert r.screenshot_base64 is None
        assert r.status == ScrapeStatus.COMPLETED
        assert r.error is None
        assert r.scraper_used is None
        assert r.processing_time_seconds == 0.0


# ─── Health Router Edge Cases ───────────────────────────────────────────────


class TestHealthEdgeCasesE2E:
    """Health endpoint edge cases."""

    @pytest.mark.asyncio
    async def test_health_live_has_timestamp(self, async_client):
        """Liveness probe includes ISO timestamp."""
        resp = await async_client.get("/health/live")
        data = resp.json()
        assert "timestamp" in data
        assert "T" in data["timestamp"]  # ISO format contains T

    @pytest.mark.asyncio
    async def test_health_live_has_service_name(self, async_client):
        """Liveness probe includes service name."""
        resp = await async_client.get("/health/live")
        data = resp.json()
        assert "service" in data
        assert data["service"] == "webscraper-service"

    @pytest.mark.asyncio
    async def test_health_ready_checks_playwright(self, async_client):
        """Readiness probe checks playwright_available."""
        resp = await async_client.get("/health/ready")
        data = resp.json()
        assert "checks" in data
        assert "playwright_available" in data["checks"]
        assert "browser_ready" in data["checks"]

    @pytest.mark.asyncio
    async def test_health_full_includes_version(self, async_client):
        """Health check includes version info."""
        resp = await async_client.get("/health")
        data = resp.json()
        assert "version" in data
        assert data["version"]  # Non-empty


# ─── Playwright Image Download ──────────────────────────────────────────────


class TestImageDownloadSharedE2E:
    """Shared image_utils download and enrichment."""

    @pytest.mark.asyncio
    async def test_download_image_success(self):
        """download_image downloads and returns image data."""
        from services.image_utils import download_image

        png_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100

        # download_image streams the body (client.stream, bounded by max_bytes)
        # since #38 (7c51161) — drive it through a real httpx client.
        transport = httpx.MockTransport(
            lambda req: httpx.Response(
                200, headers={"content-type": "image/png"}, content=png_bytes
            )
        )
        async with httpx.AsyncClient(transport=transport) as client:
            result = await download_image(client, "http://example.com/img.png")
        assert result is not None
        img_bytes, mime, src = result
        assert img_bytes == png_bytes
        assert src == "http://example.com/img.png"

    @pytest.mark.asyncio
    async def test_download_image_data_uri_decoded(self):
        """Data URIs are decoded inline."""
        import base64 as b64mod

        from services.image_utils import download_image

        raw = b"\x89PNG" + b"\x00" * 20
        b64_str = b64mod.b64encode(raw).decode()
        data_uri = f"data:image/png;base64,{b64_str}"

        mock_client = AsyncMock()
        result = await download_image(mock_client, data_uri)
        assert result is not None
        body, mime, src = result
        assert body == raw
        assert mime == "image/png"

    @pytest.mark.asyncio
    async def test_download_image_empty_src_skipped(self):
        """Empty src is skipped."""
        from services.image_utils import download_image

        mock_client = AsyncMock()
        result = await download_image(mock_client, "")
        assert result is None

    @pytest.mark.asyncio
    async def test_download_image_non_image_skipped(self):
        """Non-image content-type is skipped."""
        from services.image_utils import download_image

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "text/html"}
        mock_resp.content = b"<html></html>"
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)

        result = await download_image(mock_client, "http://example.com/page")
        assert result is None

    @pytest.mark.asyncio
    async def test_download_image_too_large_skipped(self):
        """Images larger than 5MB are skipped."""
        from services.image_utils import download_image

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "image/jpeg"}
        mock_resp.content = b"\x00" * (6 * 1024 * 1024)  # 6MB
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)

        result = await download_image(mock_client, "http://example.com/big.jpg")
        assert result is None

    @pytest.mark.asyncio
    async def test_download_image_error_returns_none(self):
        """Download errors return None."""
        from services.image_utils import download_image

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))

        result = await download_image(mock_client, "http://example.com/img.png")
        assert result is None

    @pytest.mark.asyncio
    async def test_enrich_images_with_content(self):
        """enrich_images_with_content adds base64 data to images."""
        from services.image_utils import enrich_images_with_content

        png_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 50

        images = [
            {"src": "http://example.com/img.png", "alt": "Test", "title": ""},
            {"src": "/relative.png", "alt": "Relative", "title": ""},
        ]

        async def fake_download(client, src, max_bytes=None, timeout=None):
            if src.startswith("http"):
                return png_bytes, "image/png", src
            return None

        with patch("services.image_utils.download_image", side_effect=fake_download):
            enriched = await enrich_images_with_content(images)

        assert len(enriched) == 2
        assert enriched[0]["content_base64"] is not None
        assert enriched[0]["mime_type"] is not None
        assert enriched[0]["byte_size"] == len(png_bytes)
        assert "content_base64" not in enriched[1]

    @pytest.mark.asyncio
    async def test_capture_screenshot_success(self):
        """_capture_screenshot_base64 returns base64 string."""
        from services.playwright_scraper import PlaywrightScraper

        scraper = PlaywrightScraper()
        mock_page = AsyncMock()
        mock_page.screenshot = AsyncMock(return_value=b"\x89PNG screenshot data")

        result = await scraper._capture_screenshot_base64(mock_page)
        assert result is not None
        # Verify it's valid base64
        decoded = base64.b64decode(result)
        assert decoded == b"\x89PNG screenshot data"

    @pytest.mark.asyncio
    async def test_capture_screenshot_failure(self):
        """Screenshot failure returns None."""
        from services.playwright_scraper import PlaywrightScraper

        scraper = PlaywrightScraper()
        mock_page = AsyncMock()
        mock_page.screenshot = AsyncMock(side_effect=RuntimeError("Screenshot failed"))

        result = await scraper._capture_screenshot_base64(mock_page)
        assert result is None


# ─── HttpxScraper Image Download ────────────────────────────────────────────


class TestHttpxImageDownloadE2E:
    """Shared image_utils download paths (previously on HttpxScraper)."""

    @pytest.mark.asyncio
    async def test_download_image_success(self):
        """download_image downloads and returns image data."""
        from services.image_utils import download_image

        png_bytes = b"\x89PNG data"

        # Streaming download since #38 (7c51161): real httpx client, mock transport.
        transport = httpx.MockTransport(
            lambda req: httpx.Response(
                200, headers={"content-type": "image/png"}, content=png_bytes
            )
        )
        async with httpx.AsyncClient(transport=transport) as client:
            result = await download_image(client, "http://example.com/img.png")
        assert result is not None
        assert result[0] == png_bytes

    @pytest.mark.asyncio
    async def test_download_image_data_uri_decoded(self):
        """Data URIs are decoded inline."""
        import base64 as b64mod

        from services.image_utils import download_image

        raw = b"\x89PNG" + b"\x00" * 10
        b64_str = b64mod.b64encode(raw).decode()
        data_uri = f"data:image/png;base64,{b64_str}"

        mock_client = AsyncMock()
        result = await download_image(mock_client, data_uri)
        assert result is not None
        assert result[0] == raw

    @pytest.mark.asyncio
    async def test_download_image_non_image_skipped(self):
        """Non-image content type is skipped."""
        from services.image_utils import download_image

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "application/json"}
        mock_resp.content = b"{}"
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)

        result = await download_image(mock_client, "http://example.com/api")
        assert result is None

    @pytest.mark.asyncio
    async def test_download_image_too_large(self):
        """Images >5MB are skipped."""
        from services.image_utils import download_image

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "image/jpeg"}
        mock_resp.content = b"\x00" * (6 * 1024 * 1024)
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)

        result = await download_image(mock_client, "http://example.com/huge.jpg")
        assert result is None


# ─── Router Conversion Functions ────────────────────────────────────────────


class TestRouterConversions:
    """Test scrape router conversion helpers."""

    def test_convert_scraper_type(self):
        from routers.scrape import convert_scraper_type
        from schemas.scrape import ScraperType
        from services.base_scraper import ScraperType as ServiceType

        assert convert_scraper_type(ScraperType.PLAYWRIGHT) == ServiceType.PLAYWRIGHT
        assert convert_scraper_type(ScraperType.SCRAPY) == ServiceType.SCRAPY
        assert convert_scraper_type(ScraperType.HTTPX) == ServiceType.HTTPX
        assert convert_scraper_type(ScraperType.AUTO) == ServiceType.AUTO

    def test_convert_scrape_status(self):
        from routers.scrape import convert_scrape_status
        from schemas.scrape import ScrapeStatus
        from services.base_scraper import ScrapeStatus as ServiceStatus

        assert convert_scrape_status(ServiceStatus.COMPLETED) == ScrapeStatus.COMPLETED
        assert convert_scrape_status(ServiceStatus.FAILED) == ScrapeStatus.FAILED

    def test_convert_links(self):
        from routers.scrape import _convert_links

        raw = [
            {"text": "About", "url": "http://example.com/about", "rel": None, "type": "internal"},
            {"text": "Ext", "url": "http://other.com", "rel": "nofollow", "type": "external"},
        ]
        result = _convert_links(raw)
        assert len(result) == 2
        assert result[0].text == "About"
        assert result[0].type == "internal"
        assert result[1].rel == "nofollow"

    def test_convert_images(self):
        from routers.scrape import _convert_images

        raw = [
            {
                "src": "http://example.com/img.png",
                "alt": "Logo",
                "title": "Site Logo",
                "width": 200,
                "height": 100,
                "content_base64": "abc123",
                "mime_type": "image/png",
                "byte_size": 1024,
            },
            {"src": "http://example.com/other.jpg", "alt": "", "title": ""},
        ]
        result = _convert_images(raw)
        assert len(result) == 2
        assert result[0].src == "http://example.com/img.png"
        assert result[0].content_base64 == "abc123"
        assert result[0].byte_size == 1024
        assert result[1].content_base64 is None


# ─── Additional Validation E2E ──────────────────────────────────────────────


class TestAdditionalValidationE2E:
    """Additional validation edge cases."""

    @pytest.mark.asyncio
    async def test_viewport_min_width(self, async_client, base_url):
        """Minimum viewport width (320) is accepted."""
        resp = await async_client.post(
            "/api/scrape/url",
            json={
                "url": f"{base_url}/simple",
                "scraper_type": "httpx",
                "viewport_width": 320,
            },
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_viewport_max_width(self, async_client, base_url):
        """Maximum viewport width (3840) is accepted."""
        resp = await async_client.post(
            "/api/scrape/url",
            json={
                "url": f"{base_url}/simple",
                "scraper_type": "httpx",
                "viewport_width": 3840,
            },
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_viewport_width_too_low(self, async_client, base_url):
        """Viewport width below 320 is rejected."""
        resp = await async_client.post(
            "/api/scrape/url",
            json={
                "url": f"{base_url}/simple",
                "scraper_type": "httpx",
                "viewport_width": 100,
            },
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_viewport_width_too_high(self, async_client, base_url):
        """Viewport width above 3840 is rejected."""
        resp = await async_client.post(
            "/api/scrape/url",
            json={
                "url": f"{base_url}/simple",
                "scraper_type": "httpx",
                "viewport_width": 5000,
            },
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_viewport_height_too_low(self, async_client, base_url):
        """Viewport height below 240 is rejected."""
        resp = await async_client.post(
            "/api/scrape/url",
            json={
                "url": f"{base_url}/simple",
                "scraper_type": "httpx",
                "viewport_height": 100,
            },
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_max_depth_boundary(self, async_client, base_url):
        """max_depth at boundary (5) is accepted."""
        items = [
            {
                "url": f"{base_url}/simple",
                "html_content": "<html></html>",
                "text_content": "",
                "title": "T",
                "metadata": {"status_code": 200, "final_url": f"{base_url}/simple"},
                "links": [],
                "images": [],
                "depth": 0,
            }
        ]
        with patch(_CRAWL_SEAM, new=AsyncMock(return_value=(items, []))):
            resp = await async_client.post(
                "/api/scrape/bulk",
                json={
                    "urls": [f"{base_url}/simple"],
                    "max_depth": 5,
                    "follow_links": True,
                },
            )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_max_depth_too_high(self, async_client, base_url):
        """max_depth > 5 is rejected."""
        resp = await async_client.post(
            "/api/scrape/bulk",
            json={
                "urls": [f"{base_url}/simple"],
                "max_depth": 10,
            },
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_bulk_scrape_with_allowed_domains(self, async_client, base_url):
        """allowed_domains parameter is accepted (no validation error)."""
        resp = await async_client.post(
            "/api/scrape/bulk",
            json={
                "urls": [f"{base_url}/simple"],
                "scraper_type": "httpx",
                "allowed_domains": ["127.0.0.1"],
            },
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_scrape_url_with_download_images(self, async_client, base_url):
        """download_images=true triggers image content download."""
        resp = await async_client.post(
            "/api/scrape/url",
            json={
                "url": f"{base_url}/simple",
                "scraper_type": "httpx",
                "download_images": True,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        # The local image should have been downloaded
        local_imgs = [i for i in data["images"] if "logo.png" in i["src"]]
        if local_imgs:
            assert local_imgs[0]["content_base64"] is not None


# ─── HttpxScraper Internal Methods ──────────────────────────────────────────


class TestHttpxScraperInternalsE2E:
    """Direct tests on HttpxScraper internal methods."""

    def test_extract_text(self):
        """Text extraction strips script/style/noscript.

        HttpxScraper._extract_text was replaced in #32 (8da9b19) by the shared
        services.content_extraction.extract_main_text used by every backend.
        """
        from services.content_extraction import extract_main_text
        from services.httpx_scraper import HttpxScraper

        scraper = HttpxScraper()
        html = """
        <html><body>
            <script>var x = 1;</script>
            <style>body { color: red; }</style>
            <h1>Hello</h1>
            <p>World</p>
            <noscript>No JS</noscript>
        </body></html>
        """
        assert not hasattr(scraper, "_extract_text")
        text = extract_main_text(html, "http://example.com/")
        assert "Hello" in text
        assert "World" in text
        assert "var x" not in text
        assert "color: red" not in text
        assert "No JS" not in text

    def test_extract_metadata(self):
        """_extract_metadata extracts meta tags and canonical."""
        from bs4 import BeautifulSoup

        from services.httpx_scraper import HttpxScraper

        scraper = HttpxScraper()
        html = """
        <html><head>
            <meta name="description" content="A test page">
            <meta property="og:title" content="OG Title">
            <link rel="canonical" href="http://example.com/canonical">
        </head><body></body></html>
        """
        soup = BeautifulSoup(html, "lxml")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.url = "http://example.com/page"

        # Signature is (soup, status_code, final_url, url) since #46 (8a8dadd): the
        # transport values come from whichever backend fetched (httpx or curl_cffi).
        meta = scraper._extract_metadata(
            soup, mock_resp.status_code, str(mock_resp.url), "http://example.com/page"
        )
        assert meta["description"] == "A test page"
        assert meta["og:title"] == "OG Title"
        assert meta["canonical_url"] == "http://example.com/canonical"
        assert meta["status_code"] == 200

    def test_extract_links(self):
        """_extract_links classifies internal and external links."""
        from bs4 import BeautifulSoup

        from services.httpx_scraper import HttpxScraper

        scraper = HttpxScraper()
        html = """
        <html><body>
            <a href="/about">About</a>
            <a href="http://other.com/page">External</a>
        </body></html>
        """
        soup = BeautifulSoup(html, "lxml")
        links = scraper._extract_links(soup, "http://example.com")
        assert len(links) == 2
        internal = [lnk for lnk in links if lnk["type"] == "internal"]
        external = [lnk for lnk in links if lnk["type"] == "external"]
        assert len(internal) == 1
        assert internal[0]["text"] == "About"
        assert len(external) == 1

    def test_extract_images(self):
        """_extract_images resolves relative URLs."""
        from bs4 import BeautifulSoup

        from services.httpx_scraper import HttpxScraper

        scraper = HttpxScraper()
        html = """
        <html><body>
            <img src="/images/logo.png" alt="Logo" title="Site Logo">
            <img src="http://other.com/photo.jpg" alt="External">
        </body></html>
        """
        soup = BeautifulSoup(html, "lxml")
        images = scraper._extract_images(soup, "http://example.com")
        assert len(images) == 2
        assert images[0]["src"] == "http://example.com/images/logo.png"
        assert images[0]["alt"] == "Logo"
        assert images[1]["src"] == "http://other.com/photo.jpg"

    @pytest.mark.asyncio
    async def test_httpx_scraper_close(self):
        """close() is a no-op: there is no cached client to close.

        Since #30 (71761da) each scrape builds a request-local client via
        _build_client and closes it with its own context manager, so close()
        holds nothing and a scrape after close() still works.
        """
        from services.base_scraper import ScrapeConfig
        from services.httpx_scraper import HttpxScraper

        scraper = HttpxScraper()
        cfg = ScrapeConfig()
        c1 = scraper._build_client(cfg, 5.0)
        c2 = scraper._build_client(cfg, 5.0)
        try:
            assert c1 is not c2  # request-local, never shared
        finally:
            await c1.aclose()
            await c2.aclose()
        await scraper.close()
        assert not hasattr(scraper, "_client")

    @pytest.mark.asyncio
    async def test_httpx_scraper_close_no_client(self):
        """close() is safe when no client exists."""
        from services.httpx_scraper import HttpxScraper

        scraper = HttpxScraper()
        await scraper.close()  # Should not raise


# ─── Scraper Type Enums ─────────────────────────────────────────────────────


class TestEnums:
    """StrEnum coverage for ScraperType and ScrapeStatus."""

    def test_scraper_type_values(self):
        from services.base_scraper import ScraperType

        assert ScraperType.PLAYWRIGHT == "playwright"
        assert ScraperType.SCRAPY == "scrapy"
        assert ScraperType.HTTPX == "httpx"
        assert ScraperType.AUTO == "auto"

    def test_scrape_status_values(self):
        from services.base_scraper import ScrapeStatus

        assert ScrapeStatus.QUEUED == "queued"
        assert ScrapeStatus.RUNNING == "running"
        assert ScrapeStatus.COMPLETED == "completed"
        assert ScrapeStatus.FAILED == "failed"
        assert ScrapeStatus.PAUSED == "paused"
        assert ScrapeStatus.CANCELLED == "cancelled"

    def test_schema_scraper_type_values(self):
        from schemas.scrape import ScraperType

        assert ScraperType.PLAYWRIGHT == "playwright"
        assert ScraperType.AUTO == "auto"

    def test_schema_scrape_status_values(self):
        from schemas.scrape import ScrapeStatus

        assert ScrapeStatus.COMPLETED == "completed"
        assert ScrapeStatus.FAILED == "failed"


# ─── Scrapy Internal Classes ────────────────────────────────────────────────


class TestScrapyInternals:
    """Direct tests on Scrapy spider and collector classes (no Twisted reactor)."""

    def test_result_collector_init(self):
        """_ResultCollector initialises with empty lists."""
        from services.scrapy_scraper import _ResultCollector

        c = _ResultCollector()
        assert c.items == []
        assert c.errors == []

    def test_result_collector_item_scraped(self):
        """item_scraped appends a copy of the item."""
        from services.scrapy_scraper import _ResultCollector

        c = _ResultCollector()
        item = {"url": "http://x.com", "title": "X"}
        c.item_scraped(item, MagicMock(), MagicMock())
        assert len(c.items) == 1
        assert c.items[0] == item
        # Must be a copy
        item["title"] = "changed"
        assert c.items[0]["title"] == "X"

    def test_result_collector_spider_error(self):
        """spider_error appends the stringified failure."""
        from services.scrapy_scraper import _ResultCollector

        c = _ResultCollector()
        c.spider_error("some failure", MagicMock(), MagicMock())
        assert c.errors == ["some failure"]

    def test_web_spider_init(self):
        """WebSpider stores config and parses base domain."""
        from services.base_scraper import ScrapeConfig
        from services.scrapy_scraper import WebSpider

        config = ScrapeConfig(timeout=10)
        spider = WebSpider(start_url="http://example.com/page", config=config)
        assert spider.start_urls == ["http://example.com/page"]
        assert spider._base_domain == "example.com"
        assert spider._visited == set()
        assert spider._current_depth == {"http://example.com/page": 0}

    def test_web_spider_start_no_ua(self):
        """start() yields Request without User-Agent if not configured."""
        from services.base_scraper import ScrapeConfig
        from services.scrapy_scraper import WebSpider

        config = ScrapeConfig(user_agent=None, headers={}, cookies={})
        spider = WebSpider(start_url="http://example.com", config=config)
        reqs = asyncio.run(_drain(spider.start()))
        assert len(reqs) == 1
        assert reqs[0].url == "http://example.com"
        assert reqs[0].dont_filter is True

    def test_web_spider_start_with_ua_headers_cookies(self):
        """start() applies UA, custom headers, and cookies."""
        from services.base_scraper import ScrapeConfig
        from services.scrapy_scraper import WebSpider

        config = ScrapeConfig(
            user_agent="Bot/1.0",
            headers={"X-Custom": "val"},
            cookies={"sid": "abc"},
        )
        spider = WebSpider(start_url="http://example.com", config=config)
        reqs = asyncio.run(_drain(spider.start()))
        assert len(reqs) == 1
        req = reqs[0]
        assert req.headers.get(b"User-Agent") == b"Bot/1.0"
        assert req.headers.get(b"X-Custom") == b"val"
        assert req.cookies == {"sid": "abc"}

    def test_web_spider_parse_basic(self):
        """parse extracts title, text, metadata, links, images from response."""
        from scrapy.http import HtmlResponse

        from services.base_scraper import ScrapeConfig
        from services.scrapy_scraper import WebSpider

        config = ScrapeConfig()
        spider = WebSpider(start_url="http://example.com", config=config)

        body = b"""<!DOCTYPE html>
        <html>
        <head>
            <title>Test Title</title>
            <meta name="description" content="A description">
            <meta property="og:title" content="OG Title">
            <link rel="canonical" href="http://example.com/canonical">
        </head>
        <body>
            <h1>Hello World</h1>
            <p>Some text content here.</p>
            <a href="/about">About</a>
            <a href="http://other.com/ext">External</a>
            <img src="/img.png" alt="Pic" title="Pic Title">
        </body>
        </html>"""

        response = HtmlResponse(url="http://example.com", body=body)
        items = list(spider.parse_page(response))

        assert len(items) >= 1
        item = items[0]
        assert item["title"] == "Test Title"
        assert "Hello World" in item["text_content"]
        assert "Some text content" in item["text_content"]
        assert item["metadata"]["status_code"] == 200
        assert item["metadata"]["description"] == "A description"
        assert item["metadata"]["og:title"] == "OG Title"
        assert item["metadata"]["canonical_url"] == "http://example.com/canonical"

        internal = [lnk for lnk in item["links"] if lnk["type"] == "internal"]
        external = [lnk for lnk in item["links"] if lnk["type"] == "external"]
        assert len(internal) == 1
        assert internal[0]["text"] == "About"
        assert len(external) == 1

        assert len(item["images"]) == 1
        assert item["images"][0]["alt"] == "Pic"
        assert item["images"][0]["title"] == "Pic Title"

    def test_web_spider_parse_skips_visited(self):
        """parse returns nothing for already-visited URLs."""
        from scrapy.http import HtmlResponse

        from services.base_scraper import ScrapeConfig
        from services.scrapy_scraper import WebSpider

        config = ScrapeConfig()
        spider = WebSpider(start_url="http://example.com", config=config)
        spider._visited.add("http://example.com")

        response = HtmlResponse(url="http://example.com", body=b"<html><body>X</body></html>")
        items = list(spider.parse_page(response))
        assert items == []

    def test_web_spider_parse_follows_links(self):
        """parse yields follow-up Requests when follow_links=True."""
        import scrapy
        from scrapy.http import HtmlResponse

        from services.base_scraper import ScrapeConfig
        from services.scrapy_scraper import WebSpider

        config = ScrapeConfig(follow_links=True, max_depth=2)
        spider = WebSpider(start_url="http://example.com", config=config)

        body = b"""<html><body>
            <a href="/page2">Page 2</a>
            <a href="http://other.com/ext">External</a>
        </body></html>"""

        response = HtmlResponse(url="http://example.com", body=body)
        results = list(spider.parse_page(response))

        # First result is the item dict, then follow-up requests
        item = results[0]
        assert isinstance(item, dict)

        follow_reqs = [r for r in results if isinstance(r, scrapy.Request)]
        assert len(follow_reqs) == 1  # Only internal link
        assert "page2" in follow_reqs[0].url

    def test_web_spider_parse_respects_max_depth(self):
        """parse does NOT follow links when at max_depth."""
        import scrapy
        from scrapy.http import HtmlResponse

        from services.base_scraper import ScrapeConfig
        from services.scrapy_scraper import WebSpider

        config = ScrapeConfig(follow_links=True, max_depth=1)
        spider = WebSpider(start_url="http://example.com", config=config)
        # Simulate being at depth 1 already
        spider._current_depth["http://example.com/deep"] = 1

        body = b"""<html><body>
            <a href="/page3">Page 3</a>
        </body></html>"""

        response = HtmlResponse(url="http://example.com/deep", body=body)
        results = list(spider.parse_page(response))

        follow_reqs = [r for r in results if isinstance(r, scrapy.Request)]
        assert len(follow_reqs) == 0  # At max depth, no following

    def test_web_spider_parse_no_follow_when_disabled(self):
        """parse does NOT follow links when follow_links=False."""
        import scrapy
        from scrapy.http import HtmlResponse

        from services.base_scraper import ScrapeConfig
        from services.scrapy_scraper import WebSpider

        config = ScrapeConfig(follow_links=False)
        spider = WebSpider(start_url="http://example.com", config=config)

        body = b'<html><body><a href="/page2">P</a></body></html>'
        response = HtmlResponse(url="http://example.com", body=body)
        results = list(spider.parse_page(response))

        follow_reqs = [r for r in results if isinstance(r, scrapy.Request)]
        assert len(follow_reqs) == 0

    def test_web_spider_errback(self):
        """errback logs error without raising."""
        from services.base_scraper import ScrapeConfig
        from services.scrapy_scraper import WebSpider

        config = ScrapeConfig()
        spider = WebSpider(start_url="http://example.com", config=config)
        spider.errback("Some failure message")  # Should not raise

    def test_run_crawl_settings_config(self):
        """_run_crawl configures Scrapy settings correctly."""
        from services.base_scraper import ScrapeConfig
        from services.scrapy_scraper import _run_crawl

        config = ScrapeConfig(
            timeout=15,
            max_retries=2,
            delay=0.5,
            follow_links=True,
            max_depth=3,
            respect_robots_txt=False,
            user_agent="TestBot/2.0",
        )

        # Mock CrawlerProcess to capture settings without starting Twisted
        with patch("services.scrapy_scraper.CrawlerProcess") as mock_proc_cls:
            mock_process = MagicMock()
            mock_process.crawl = MagicMock()
            mock_process.start = MagicMock()
            mock_proc_cls.return_value = mock_process

            with patch("services.scrapy_scraper.dispatcher"):
                _run_crawl("http://example.com", config)

            # Verify settings passed to CrawlerProcess
            call_args = mock_proc_cls.call_args
            settings = call_args[0][0]
            assert settings.get("DOWNLOAD_TIMEOUT") == 15
            assert settings.get("RETRY_TIMES") == 2
            assert settings.get("DOWNLOAD_DELAY") == 0.5
            assert settings.get("DEPTH_LIMIT") == 3
            assert settings.get("ROBOTSTXT_OBEY") is False
            assert settings.get("USER_AGENT") == "TestBot/2.0"
            assert settings.get("LOG_ENABLED") is False
            assert settings.get("TELNETCONSOLE_ENABLED") is False

    def test_run_crawl_depth_limit_zero_when_no_follow(self):
        """_run_crawl sets DEPTH_LIMIT=0 when follow_links=False."""
        from services.base_scraper import ScrapeConfig
        from services.scrapy_scraper import _run_crawl

        config = ScrapeConfig(follow_links=False, max_depth=5)

        with patch("services.scrapy_scraper.CrawlerProcess") as mock_proc_cls:
            mock_process = MagicMock()
            mock_process.crawl = MagicMock()
            mock_process.start = MagicMock()
            mock_proc_cls.return_value = mock_process

            with patch("services.scrapy_scraper.dispatcher"):
                _run_crawl("http://example.com", config)

            settings = mock_proc_cls.call_args[0][0]
            assert settings.get("DEPTH_LIMIT") == 0

    def test_run_crawl_no_user_agent(self):
        """_run_crawl does not explicitly set USER_AGENT when none configured."""
        from services.base_scraper import ScrapeConfig
        from services.scrapy_scraper import _run_crawl

        config = ScrapeConfig(user_agent=None)

        with patch("services.scrapy_scraper.CrawlerProcess") as mock_proc_cls:
            mock_process = MagicMock()
            mock_process.crawl = MagicMock()
            mock_process.start = MagicMock()
            mock_proc_cls.return_value = mock_process

            with patch("services.scrapy_scraper.dispatcher"):
                _run_crawl("http://example.com", config)

            settings = mock_proc_cls.call_args[0][0]
            # When user_agent is None, we don't call settings.set("USER_AGENT", ...)
            # so it retains Scrapy's built-in default
            assert "TestBot" not in settings.get("USER_AGENT", "")

    def test_run_crawl_connects_signals(self):
        """_run_crawl connects item_scraped and spider_error signals."""
        from services.base_scraper import ScrapeConfig
        from services.scrapy_scraper import _run_crawl

        config = ScrapeConfig()

        with patch("services.scrapy_scraper.CrawlerProcess") as mock_proc_cls:
            mock_process = MagicMock()
            mock_process.crawl = MagicMock()
            mock_process.start = MagicMock()
            mock_proc_cls.return_value = mock_process

            with patch("services.scrapy_scraper.dispatcher") as mock_disp:
                _run_crawl("http://example.com", config)

                assert mock_disp.connect.call_count == 2


# ─── ScraperManager Remaining Gaps ──────────────────────────────────────────


class TestScraperManagerGaps:
    """Cover remaining ScraperManager branches."""

    @pytest.mark.asyncio
    async def test_auto_selects_playwright_for_javascript_enabled(self, async_client, base_url):
        """AUTO mode picks Playwright when javascript_enabled=True in config."""
        from services.base_scraper import ScrapeResult, ScrapeStatus

        with patch(
            "services.playwright_scraper.PlaywrightScraper.scrape",
            return_value=ScrapeResult(
                url=f"{base_url}/simple",
                html_content="<html><body>JS</body></html>",
                text_content="JS",
                title="JS Page",
                metadata={"status_code": 200, "final_url": f"{base_url}/simple"},
                status=ScrapeStatus.COMPLETED,
                scraper_used="PlaywrightScraper",
                processing_time_seconds=0.1,
            ),
        ):
            # Need to set javascript_enabled on config. This happens through
            # ScraperManager._select_auto_scraper when config.javascript_enabled=True.
            # We test the manager directly since the API schema doesn't expose javascript_enabled.
            from services.base_scraper import ScrapeConfig
            from services.scraper_manager import ScraperManager

            mgr = ScraperManager()
            config = ScrapeConfig(javascript_enabled=True)
            scraper = await mgr._select_auto_scraper(f"{base_url}/simple", config)
            assert scraper.name == "PlaywrightScraper"

    @pytest.mark.asyncio
    async def test_auto_selects_httpx_for_other_status(self):
        """AUTO mode picks httpx for non-standard status codes (e.g., 301)."""
        from services.base_scraper import ScrapeConfig
        from services.scraper_manager import ScraperManager

        mgr = ScraperManager()
        config = ScrapeConfig()

        with patch.object(mgr, "_head_check", return_value=301):
            scraper = await mgr._select_auto_scraper("http://example.com", config)
        assert scraper.name == "HttpxScraper"

    @pytest.mark.asyncio
    async def test_scrape_url_default_config(self):
        """scrape_url creates default config when none provided."""
        from services.base_scraper import ScrapeResult, ScrapeStatus
        from services.scraper_manager import ScraperManager

        mgr = ScraperManager()
        mock_scraper = AsyncMock()
        mock_scraper.name = "HttpxScraper"
        mock_scraper.scrape = AsyncMock(
            return_value=ScrapeResult(
                url="http://example.com",
                status=ScrapeStatus.COMPLETED,
                scraper_used="HttpxScraper",
                processing_time_seconds=0.1,
            )
        )

        with patch.object(mgr, "_get_scraper", return_value=mock_scraper):
            result = await mgr.scrape_url(
                "http://example.com",
                scraper_type=ScraperManager.__module__
                and __import__("services.base_scraper", fromlist=["ScraperType"]).ScraperType.HTTPX,
                config=None,
            )
        assert result.status == ScrapeStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_scrape_url_kwargs_config_overrides(self):
        """kwargs override config fields (timeout, screenshot, etc.)."""
        from services.base_scraper import ScrapeConfig, ScrapeResult, ScraperType, ScrapeStatus
        from services.scraper_manager import ScraperManager

        mgr = ScraperManager()
        mock_scraper = AsyncMock()
        mock_scraper.name = "HttpxScraper"

        async def fake_scrape(url, **kwargs):
            return ScrapeResult(
                url=url,
                status=ScrapeStatus.COMPLETED,
                scraper_used="HttpxScraper",
                processing_time_seconds=0.1,
            )

        mock_scraper.scrape = fake_scrape

        with patch.object(mgr, "_get_scraper", return_value=mock_scraper):
            result = await mgr.scrape_url(
                "http://example.com",
                scraper_type=ScraperType.HTTPX,
                config=ScrapeConfig(),
                timeout=99.0,
                screenshot=True,
                headers={"X-Test": "1"},
                cookies={"k": "v"},
                user_agent="Override/1.0",
            )
        assert result.status == ScrapeStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_scrape_url_exception_handler(self):
        """scrape_url catches exceptions and returns FAILED."""
        from services.base_scraper import ScrapeConfig, ScraperType, ScrapeStatus
        from services.scraper_manager import ScraperManager

        mgr = ScraperManager()

        with patch.object(mgr, "_get_scraper", side_effect=RuntimeError("Unexpected error")):
            result = await mgr.scrape_url(
                "http://example.com",
                scraper_type=ScraperType.HTTPX,
                config=ScrapeConfig(),
            )
        assert result.status == ScrapeStatus.FAILED
        assert "RuntimeError" in result.error

    @pytest.mark.asyncio
    async def test_scrape_bulk_exception_handler(self):
        """scrape_bulk catches per-URL exceptions without stopping."""
        from services.base_scraper import ScrapeConfig, ScraperType, ScrapeStatus
        from services.scraper_manager import ScraperManager

        mgr = ScraperManager()
        call_count = 0

        async def flaky_scrape(url, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Boom")
            from services.base_scraper import ScrapeResult

            return ScrapeResult(
                url=url,
                status=ScrapeStatus.COMPLETED,
                scraper_used="HttpxScraper",
                processing_time_seconds=0.1,
            )

        with patch.object(mgr, "scrape_url", side_effect=flaky_scrape):
            results = await mgr.scrape_bulk(
                ["http://a.com", "http://b.com"],
                scraper_type=ScraperType.HTTPX,
                config=ScrapeConfig(),
            )
        assert len(results) == 2
        assert results[0].status == ScrapeStatus.FAILED
        assert results[1].status == ScrapeStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_get_scraper_auto_defaults_httpx(self):
        """_get_scraper with AUTO type returns httpx (shouldn't reach here normally)."""
        from services.base_scraper import ScrapeConfig, ScraperType
        from services.scraper_manager import ScraperManager

        mgr = ScraperManager()
        scraper = await mgr._get_scraper(ScraperType.AUTO, ScrapeConfig())
        assert scraper.name == "HttpxScraper"


# ─── Health Router Remaining Gaps ───────────────────────────────────────────


class TestHealthRouterGaps:
    """Cover health endpoint exception/unavailable branches."""

    @pytest.mark.asyncio
    async def test_readiness_playwright_unavailable(self, async_client):
        """Readiness returns 503 when playwright is not available."""
        with patch("importlib.util.find_spec", return_value=None):
            resp = await async_client.get("/health/ready")
        # When playwright unavailable, is_ready=False → 503
        assert resp.status_code == 503
        data = resp.json()
        assert data["status"] == "not_ready"
        assert data["checks"]["playwright_available"] is False

    @pytest.mark.asyncio
    async def test_readiness_exception_returns_503(self, async_client):
        """Readiness probe returns 503 on unexpected exception."""
        with patch(
            "routers.health.get_scraper_manager",
            side_effect=RuntimeError("Manager broken"),
        ):
            resp = await async_client.get("/health/ready")
        assert resp.status_code == 503
        data = resp.json()
        assert data["status"] == "not_ready"
        assert "error" in data

    @pytest.mark.asyncio
    async def test_health_check_exception_returns_unhealthy(self, async_client):
        """Health check returns unhealthy on exception."""
        with patch(
            "routers.health.get_scraper_manager",
            side_effect=RuntimeError("Broken"),
        ):
            resp = await async_client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "unhealthy"
        assert data["playwright_available"] is False


# ─── Main App Remaining Gaps ────────────────────────────────────────────────


class TestMainAppGaps:
    """Cover main.py lifespan and OpenAPI caching."""

    @pytest.mark.asyncio
    async def test_lifespan_startup_shutdown(self):
        """Lifespan context manager initializes and closes scraper manager."""
        from fastapi import FastAPI

        import services.scraper_manager as sm
        from main import lifespan

        old = sm._scraper_manager
        sm._scraper_manager = None

        test_app = FastAPI()

        async with lifespan(test_app):
            # During startup, scraper manager should exist
            assert sm._scraper_manager is not None

        # After shutdown, manager should be closed
        sm._scraper_manager = old

    @pytest.mark.asyncio
    async def test_openapi_schema_caching(self, async_client):
        """Second call to custom_openapi returns cached schema."""
        from main import app, custom_openapi

        # Clear any cached schema
        app.openapi_schema = None

        # First call generates schema
        schema1 = custom_openapi()
        assert schema1 is not None

        # Second call returns cached
        schema2 = custom_openapi()
        assert schema2 is schema1

        # Clean up
        app.openapi_schema = None

    @pytest.mark.asyncio
    async def test_scrape_router_exception_returns_500(self, async_client, base_url):
        """Router exception handler returns HTTP 500."""
        # Need to patch the scraper_manager's scrape_url to raise AFTER auth passes
        from services.scraper_manager import ScraperManager

        with patch.object(
            ScraperManager,
            "scrape_url",
            side_effect=RuntimeError("Manager explosion"),
        ):
            resp = await async_client.post(
                "/api/scrape/url",
                json={"url": f"{base_url}/simple", "scraper_type": "httpx"},
            )
        assert resp.status_code == 500
        # Standard error envelope since #14 (318b7c5, main.py:_error_envelope).
        err = resp.json()["error"]
        assert err["code"] == "HTTP_500"
        assert "Scraping failed" in err["message"]

    @pytest.mark.asyncio
    async def test_bulk_scrape_router_exception_returns_500(self, async_client, base_url):
        """Bulk router exception handler returns HTTP 500."""
        from services.scraper_manager import ScraperManager

        with patch.object(
            ScraperManager,
            "scrape_bulk",
            side_effect=RuntimeError("Bulk explosion"),
        ):
            resp = await async_client.post(
                "/api/scrape/bulk",
                json={"urls": [f"{base_url}/simple"], "scraper_type": "httpx"},
            )
        assert resp.status_code == 500
        # Standard error envelope since #14 (318b7c5, main.py:_error_envelope).
        err = resp.json()["error"]
        assert err["code"] == "HTTP_500"
        assert "Bulk scraping failed" in err["message"]


# ─── Auth Middleware Remaining Gaps ──────────────────────────────────────────


class TestAuthMiddlewareGaps:
    """Cover remaining auth middleware branches."""

    @pytest.mark.asyncio
    async def test_bypass_auth_via_http(self):
        """bypass_auth=True sets test_user and allows request through."""
        from fastapi import FastAPI

        from auth.middleware import AuthMiddleware

        bypass_app = FastAPI()

        @bypass_app.get("/api/test")
        async def test_endpoint():
            return {"ok": True}

        bypass_app.add_middleware(
            AuthMiddleware,
            protected_prefixes=["/api"],
            bypass_auth=True,
        )

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=bypass_app),
            base_url="http://testserver",
        ) as client:
            resp = await client.get("/api/test")

        assert resp.status_code == 200
        assert resp.json() == {"ok": True}

    @pytest.mark.asyncio
    async def test_middleware_internal_error(self):
        """Middleware exception returns 500 AUTH_ERROR."""
        from fastapi import FastAPI

        from auth.middleware import AuthMiddleware

        err_app = FastAPI()

        @err_app.get("/api/boom")
        async def boom_endpoint():
            return {"ok": True}

        err_app.add_middleware(
            AuthMiddleware,
            protected_prefixes=["/api"],
        )

        # Force dispatch to raise by patching _is_public_path to throw
        with patch.object(
            AuthMiddleware,
            "_is_public_path",
            side_effect=RuntimeError("Kaboom"),
        ):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=err_app),
                base_url="http://testserver",
            ) as client:
                resp = await client.get("/api/boom")

        assert resp.status_code == 500
        assert resp.json()["error"]["code"] == "AUTH_ERROR"

    @pytest.mark.asyncio
    async def test_introspect_license_non_200(self):
        """5xx is an availability failure → raise (→ 503); 4xx → None (→ 401)."""
        from auth.middleware import (
            LicenseServiceUnavailableError,
            _introspect_license,
            _license_cache,
        )

        _license_cache.clear()

        mock_resp = MagicMock()
        mock_resp.status_code = 500

        with patch("auth.middleware.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock()
            mock_cls.return_value = mock_client

            with pytest.raises(LicenseServiceUnavailableError):
                await _introspect_license("error-status-token")

        mock_resp_4xx = MagicMock()
        mock_resp_4xx.status_code = 403
        with patch("auth.middleware.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_resp_4xx)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock()
            mock_cls.return_value = mock_client

            assert await _introspect_license("bad-token") is None
        _license_cache.clear()

    @pytest.mark.asyncio
    async def test_extract_license_token_no_bearer_prefix(self):
        """X-Tenant-Id without 'Bearer ' prefix returns None."""
        from auth.middleware import AuthMiddleware

        middleware = AuthMiddleware(MagicMock(), protected_prefixes=["/api"])
        mock_request = MagicMock()
        mock_request.headers = {"X-Tenant-Id": "plain-token-no-bearer"}
        result = middleware._extract_license_token(mock_request)
        assert result is None

    @pytest.mark.asyncio
    async def test_extract_license_token_bearer_case_insensitive(self):
        """Bearer extraction is case-insensitive."""
        from auth.middleware import AuthMiddleware

        middleware = AuthMiddleware(MagicMock(), protected_prefixes=["/api"])
        mock_request = MagicMock()
        mock_request.headers = {"X-Tenant-Id": "BEARER my-token"}
        result = middleware._extract_license_token(mock_request)
        assert result == "my-token"

    @pytest.mark.asyncio
    async def test_extract_license_token_missing_header(self):
        """Missing X-Tenant-Id header returns None."""
        from auth.middleware import AuthMiddleware

        middleware = AuthMiddleware(MagicMock(), protected_prefixes=["/api"])
        mock_request = MagicMock()
        mock_request.headers = {}
        result = middleware._extract_license_token(mock_request)
        assert result is None

    def test_is_public_path_exact_match(self):
        """Public paths matched exactly."""
        from auth.middleware import AuthMiddleware

        middleware = AuthMiddleware(MagicMock(), protected_prefixes=["/api"])
        assert middleware._is_public_path("/") is True
        assert middleware._is_public_path("/health") is True
        assert middleware._is_public_path("/docs") is True
        assert middleware._is_public_path("/openapi.json") is True

    def test_is_public_path_prefix_match(self):
        """Public path prefix matching (e.g., /health/live)."""
        from auth.middleware import AuthMiddleware

        middleware = AuthMiddleware(MagicMock(), protected_prefixes=["/api"])
        assert middleware._is_public_path("/health/live") is True
        assert middleware._is_public_path("/health/ready") is True
        assert middleware._is_public_path("/docs/something") is True
        assert middleware._is_public_path("/static/style.css") is True

    def test_is_public_path_non_public(self):
        """Non-public paths return False."""
        from auth.middleware import AuthMiddleware

        middleware = AuthMiddleware(MagicMock(), protected_prefixes=["/api"])
        assert middleware._is_public_path("/api/scrape") is False
        assert middleware._is_public_path("/api") is False
        assert middleware._is_public_path("/random") is False

    def test_is_protected_path(self):
        """Protected path matching."""
        from auth.middleware import AuthMiddleware

        middleware = AuthMiddleware(MagicMock(), protected_prefixes=["/api"])
        assert middleware._is_protected_path("/api") is True
        assert middleware._is_protected_path("/api/scrape/url") is True
        assert middleware._is_protected_path("/health") is False
        assert middleware._is_protected_path("/") is False


# ─── Playwright Scraper Remaining Gaps ──────────────────────────────────────


class TestPlaywrightScraperGaps:
    """Cover remaining PlaywrightScraper branches.

    page.evaluate side_effects follow the CURRENT call order — metadata, links,
    images. text_content stopped being an innerText evaluate in #32 (8da9b19);
    it is extract_main_text(page.content()).
    """

    @pytest.mark.asyncio
    async def test_is_initialized_false_by_default(self):
        """New PlaywrightScraper is not initialized."""
        from services.playwright_scraper import PlaywrightScraper

        scraper = PlaywrightScraper()
        assert scraper.is_initialized() is False

    @pytest.mark.asyncio
    async def test_scrape_with_download_images(self):
        """Scrape with download_images triggers _enrich_images_with_content."""
        from services.playwright_scraper import PlaywrightScraper

        scraper = PlaywrightScraper()

        mock_page = AsyncMock()
        mock_page.goto = AsyncMock(return_value=MagicMock(status=200))
        mock_page.content = AsyncMock(return_value="<html><body>X</body></html>")
        mock_page.title = AsyncMock(return_value="T")
        mock_page.url = "http://example.com"
        mock_page.evaluate = AsyncMock(
            side_effect=[
                {},
                [],
                [
                    {
                        "src": "http://example.com/img.png",
                        "alt": "",
                        "title": "",
                        "width": 0,
                        "height": 0,
                    }
                ],
            ]
        )
        mock_page.close = AsyncMock()
        mock_page.context = MagicMock()

        mock_browser = AsyncMock()
        mock_browser.new_page = AsyncMock(return_value=mock_page)

        scraper.browser = mock_browser
        scraper._initialized = True

        with patch(
            "services.playwright_scraper.enrich_images_with_content",
            return_value=[
                {
                    "src": "http://example.com/img.png",
                    "alt": "",
                    "title": "",
                    "content_base64": "abc",
                }
            ],
        ) as mock_enrich:
            result = await scraper.scrape("http://example.com", download_images=True)

        mock_enrich.assert_awaited_once()
        assert result.images[0]["content_base64"] == "abc"

    @pytest.mark.asyncio
    async def test_scrape_invalid_cookies_type(self):
        """Non-dict cookies are warned and skipped."""
        from services.playwright_scraper import PlaywrightScraper

        scraper = PlaywrightScraper()

        mock_page = AsyncMock()
        mock_page.goto = AsyncMock(return_value=MagicMock(status=200))
        mock_page.content = AsyncMock(return_value="<html></html>")
        mock_page.title = AsyncMock(return_value="T")
        mock_page.url = "http://example.com"
        mock_page.evaluate = AsyncMock(side_effect=[{}, [], []])
        mock_page.close = AsyncMock()
        mock_page.context = MagicMock()

        mock_browser = AsyncMock()
        mock_browser.new_page = AsyncMock(return_value=mock_page)

        scraper.browser = mock_browser
        scraper._initialized = True
        scraper.config.cookies = "not-a-dict"

        result = await scraper.scrape("http://example.com")
        assert result.status.value == "completed"  # Should still succeed

    @pytest.mark.asyncio
    async def test_metadata_extraction_error(self):
        """Metadata extraction error returns empty dict."""
        from services.playwright_scraper import PlaywrightScraper

        scraper = PlaywrightScraper()

        mock_page = AsyncMock()
        mock_page.goto = AsyncMock(return_value=MagicMock(status=200))
        mock_page.content = AsyncMock(return_value="<html></html>")
        mock_page.title = AsyncMock(return_value="T")
        mock_page.url = "http://example.com"
        mock_page.evaluate = AsyncMock(
            side_effect=[
                RuntimeError("JS error"),  # metadata extraction fails
                [],  # links
                [],  # images
            ]
        )
        mock_page.close = AsyncMock()
        mock_page.context = MagicMock()

        mock_browser = AsyncMock()
        mock_browser.new_page = AsyncMock(return_value=mock_page)

        scraper.browser = mock_browser
        scraper._initialized = True

        result = await scraper.scrape("http://example.com")
        assert result.status.value == "completed"
        # Metadata should have status_code and final_url even if JS extraction failed
        assert "final_url" in result.metadata

    @pytest.mark.asyncio
    async def test_links_extraction_error(self):
        """Links extraction error returns empty list."""
        from services.playwright_scraper import PlaywrightScraper

        scraper = PlaywrightScraper()

        mock_page = AsyncMock()
        mock_page.goto = AsyncMock(return_value=MagicMock(status=200))
        mock_page.content = AsyncMock(return_value="<html></html>")
        mock_page.title = AsyncMock(return_value="T")
        mock_page.url = "http://example.com"
        mock_page.evaluate = AsyncMock(
            side_effect=[
                {"description": "test"},  # metadata OK
                RuntimeError("Links JS error"),  # links fail
                [],  # images
            ]
        )
        mock_page.close = AsyncMock()
        mock_page.context = MagicMock()

        mock_browser = AsyncMock()
        mock_browser.new_page = AsyncMock(return_value=mock_page)

        scraper.browser = mock_browser
        scraper._initialized = True

        result = await scraper.scrape("http://example.com")
        assert result.status.value == "completed"
        assert result.links == []

    @pytest.mark.asyncio
    async def test_images_extraction_error(self):
        """Images extraction error returns empty list."""
        from services.playwright_scraper import PlaywrightScraper

        scraper = PlaywrightScraper()

        mock_page = AsyncMock()
        mock_page.goto = AsyncMock(return_value=MagicMock(status=200))
        mock_page.content = AsyncMock(return_value="<html></html>")
        mock_page.title = AsyncMock(return_value="T")
        mock_page.url = "http://example.com"
        mock_page.evaluate = AsyncMock(
            side_effect=[
                {},
                [],  # links OK
                RuntimeError("Images JS error"),  # images fail
            ]
        )
        mock_page.close = AsyncMock()
        mock_page.context = MagicMock()

        mock_browser = AsyncMock()
        mock_browser.new_page = AsyncMock(return_value=mock_page)

        scraper.browser = mock_browser
        scraper._initialized = True

        result = await scraper.scrape("http://example.com")
        assert result.status.value == "completed"
        assert result.images == []
