"""
E2E tests for webscraper_service.

These tests exercise the full stack: HTTP request → FastAPI router → ScraperManager
→ HttpxScraper → real HTTP server → parsed response.  No mocks on the scraping path.

No real browser is needed: Playwright paths are covered with a mocked browser in
test_e2e_backends.py. Everything here runs with the lightweight httpx backend.
"""

import pytest

# ``e2e``, not ``integration``: in-process full stack (ASGI app + a local HTTP
# server, Playwright mocked), no external network and no live service, so it runs
# in the default suite and CI. ``integration`` is reserved for tests that need a
# live deployment or a real browser.
pytestmark = pytest.mark.e2e


# ─── Health Endpoints ────────────────────────────────────────────────────────


class TestHealthE2E:
    """Full-stack health endpoint tests."""

    @pytest.mark.asyncio
    async def test_root_returns_service_info(self, async_client):
        resp = await async_client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["service"] == "webscraper-service"
        assert "version" in data
        assert "endpoints" in data

    @pytest.mark.asyncio
    async def test_health_check(self, async_client):
        resp = await async_client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("healthy", "unhealthy")
        assert "timestamp" in data
        assert "playwright_available" in data
        assert "browser_ready" in data

    @pytest.mark.asyncio
    async def test_liveness_probe(self, async_client):
        resp = await async_client.get("/health/live")
        assert resp.status_code == 200
        assert resp.json()["status"] == "alive"

    @pytest.mark.asyncio
    async def test_readiness_probe(self, async_client):
        resp = await async_client.get("/health/ready")
        assert resp.status_code in (200, 503)
        data = resp.json()
        assert data["status"] in ("ready", "not_ready")
        assert "checks" in data


# ─── Single URL Scraping ─────────────────────────────────────────────────────


class TestScrapeUrlE2E:
    """POST /api/scrape/url — full request-to-response cycle."""

    @pytest.mark.asyncio
    async def test_scrape_simple_page(self, async_client, base_url):
        """Scrape a simple HTML page and verify all response fields."""
        resp = await async_client.post(
            "/api/scrape/url",
            json={"url": f"{base_url}/simple", "scraper_type": "httpx"},
        )
        assert resp.status_code == 200
        data = resp.json()

        # Status
        assert data["status"] == "completed"
        assert data["scraper_used"] == "HttpxScraper"
        assert data["processing_time_seconds"] > 0
        assert data["error"] is None

        # Content
        assert data["url"] == f"{base_url}/simple"
        assert "<h1>Hello E2E</h1>" in data["html_content"]
        assert "Hello E2E" in data["text_content"]
        assert data["title"] == "Test Page"

        # Metadata
        meta = data["metadata"]
        assert meta["status_code"] == 200
        assert "description" in meta
        assert meta["description"] == "A simple test page"
        assert "og:title" in meta
        assert "canonical_url" in meta

        # Links
        links = data["links"]
        assert len(links) == 2
        internal = [lnk for lnk in links if lnk["type"] == "internal"]
        external = [lnk for lnk in links if lnk["type"] == "external"]
        assert len(internal) == 1
        assert internal[0]["text"] == "About"
        assert len(external) == 1
        assert "external.example.com" in external[0]["url"]

        # Images (no download — just metadata)
        images = data["images"]
        assert len(images) == 2
        assert images[0]["alt"] == "Logo"
        assert images[0]["title"] == "Site Logo"
        assert images[0]["content_base64"] is None  # not requested

    @pytest.mark.asyncio
    async def test_scrape_with_image_download(self, async_client, base_url):
        """Scrape with download_images=true — local image gets base64."""
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

        images = data["images"]
        # The local /images/logo.png should be downloaded and base64-encoded
        local_img = next((i for i in images if "logo.png" in i["src"]), None)
        assert local_img is not None
        assert local_img["content_base64"] is not None
        assert len(local_img["content_base64"]) > 0
        assert local_img["mime_type"] is not None
        assert local_img["byte_size"] > 0

    @pytest.mark.asyncio
    async def test_scrape_empty_page(self, async_client, base_url):
        """Scrape a page with no content — should still succeed."""
        resp = await async_client.post(
            "/api/scrape/url",
            json={"url": f"{base_url}/empty", "scraper_type": "httpx"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        assert data["title"] == "Empty"
        assert data["links"] == []
        assert data["images"] == []

    @pytest.mark.asyncio
    async def test_scrape_page_with_links(self, async_client, base_url):
        """Scrape a link-heavy page and verify link extraction."""
        resp = await async_client.post(
            "/api/scrape/url",
            json={"url": f"{base_url}/links", "scraper_type": "httpx"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        assert len(data["links"]) == 4
        internal = [lnk for lnk in data["links"] if lnk["type"] == "internal"]
        external = [lnk for lnk in data["links"] if lnk["type"] == "external"]
        assert len(internal) == 3
        assert len(external) == 1

    @pytest.mark.asyncio
    async def test_scrape_redirect_follows(self, async_client, base_url):
        """Scraping a redirect URL follows to the target page."""
        resp = await async_client.post(
            "/api/scrape/url",
            json={"url": f"{base_url}/redirect", "scraper_type": "httpx"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        # The redirected page is /simple
        assert data["title"] == "Test Page"
        assert data["metadata"]["final_url"].endswith("/simple")

    @pytest.mark.asyncio
    async def test_scrape_server_error_returns_failed(self, async_client, base_url):
        """Scraping a 500 page returns status=failed, not a 500 from our API."""
        resp = await async_client.post(
            "/api/scrape/url",
            json={"url": f"{base_url}/error500", "scraper_type": "httpx"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "failed"
        assert "500" in data["error"]

    @pytest.mark.asyncio
    async def test_scrape_nonexistent_path_returns_failed(self, async_client, base_url):
        """Scraping a 404 page returns status=failed."""
        resp = await async_client.post(
            "/api/scrape/url",
            json={"url": f"{base_url}/no-such-page", "scraper_type": "httpx"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "failed"
        assert "404" in data["error"]

    @pytest.mark.asyncio
    async def test_scrape_timeout(self, async_client, base_url):
        """Scraping a slow page with short timeout returns failed."""
        resp = await async_client.post(
            "/api/scrape/url",
            json={
                "url": f"{base_url}/slow",
                "scraper_type": "httpx",
                "timeout": 2.0,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "failed"
        assert data["scraper_used"] == "HttpxScraper"
        assert "timeout" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_scrape_auto_selects_httpx_for_simple_page(self, async_client, base_url):
        """AUTO mode picks httpx for a page that responds to HEAD with 200."""
        resp = await async_client.post(
            "/api/scrape/url",
            json={"url": f"{base_url}/simple", "scraper_type": "auto"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        assert data["scraper_used"] == "HttpxScraper"

    @pytest.mark.asyncio
    async def test_scrape_custom_headers(self, async_client, base_url):
        """Custom headers are forwarded — doesn't break scraping."""
        resp = await async_client.post(
            "/api/scrape/url",
            json={
                "url": f"{base_url}/simple",
                "scraper_type": "httpx",
                "headers": {"X-Custom": "test-value"},
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"

    @pytest.mark.asyncio
    async def test_scrape_custom_user_agent(self, async_client, base_url):
        """Custom user agent doesn't break scraping."""
        resp = await async_client.post(
            "/api/scrape/url",
            json={
                "url": f"{base_url}/simple",
                "scraper_type": "httpx",
                "user_agent": "TestBot/1.0",
            },
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "completed"


# ─── Input Validation ────────────────────────────────────────────────────────


class TestValidationE2E:
    """Request validation at the API boundary."""

    @pytest.mark.asyncio
    async def test_invalid_url_returns_422(self, async_client):
        resp = await async_client.post(
            "/api/scrape/url",
            json={"url": "not-a-url"},
        )
        assert resp.status_code == 422
        # Standard error envelope since #14 (318b7c5, main.py:validation_exception_handler).
        err = resp.json()["error"]
        assert err["code"] == "VALIDATION_ERROR"
        assert err["status"] == 422

    @pytest.mark.asyncio
    async def test_missing_url_returns_422(self, async_client):
        resp = await async_client.post("/api/scrape/url", json={})
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_invalid_scraper_type_returns_422(self, async_client, base_url):
        resp = await async_client.post(
            "/api/scrape/url",
            json={"url": f"{base_url}/simple", "scraper_type": "nonexistent"},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_timeout_too_high_returns_422(self, async_client, base_url):
        resp = await async_client.post(
            "/api/scrape/url",
            json={"url": f"{base_url}/simple", "timeout": 999},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_timeout_too_low_returns_422(self, async_client, base_url):
        resp = await async_client.post(
            "/api/scrape/url",
            json={"url": f"{base_url}/simple", "timeout": 0.1},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_bulk_empty_urls_returns_422(self, async_client):
        resp = await async_client.post(
            "/api/scrape/bulk",
            json={"urls": []},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_bulk_too_many_urls_returns_422(self, async_client, base_url):
        resp = await async_client.post(
            "/api/scrape/bulk",
            json={"urls": [f"{base_url}/simple"] * 101},
        )
        assert resp.status_code == 422


# ─── Bulk Scraping ───────────────────────────────────────────────────────────


class TestBulkScrapeE2E:
    """POST /api/scrape/bulk — full request-to-response cycle."""

    @pytest.mark.asyncio
    async def test_bulk_scrape_multiple_urls(self, async_client, base_url):
        """Bulk scrape multiple pages — verify counts and individual results."""
        resp = await async_client.post(
            "/api/scrape/bulk",
            json={
                "urls": [f"{base_url}/simple", f"{base_url}/links", f"{base_url}/empty"],
                "scraper_type": "httpx",
            },
        )
        assert resp.status_code == 200
        data = resp.json()

        assert data["total_urls"] == 3
        assert data["completed"] == 3
        assert data["failed"] == 0
        assert data["processing_time_seconds"] > 0
        assert len(data["results"]) == 3

        # Each result is a full ScrapeResponse
        titles = {r["title"] for r in data["results"]}
        assert "Test Page" in titles
        assert "Link Page" in titles
        assert "Empty" in titles

    @pytest.mark.asyncio
    async def test_bulk_scrape_partial_failure(self, async_client, base_url):
        """Bulk scrape with one failing URL — others still succeed."""
        resp = await async_client.post(
            "/api/scrape/bulk",
            json={
                "urls": [f"{base_url}/simple", f"{base_url}/error500"],
                "scraper_type": "httpx",
            },
        )
        assert resp.status_code == 200
        data = resp.json()

        assert data["total_urls"] == 2
        assert data["completed"] == 1
        assert data["failed"] == 1

        results = data["results"]
        ok = [r for r in results if r["status"] == "completed"]
        fail = [r for r in results if r["status"] == "failed"]
        assert len(ok) == 1
        assert len(fail) == 1
        assert ok[0]["title"] == "Test Page"
        assert "500" in fail[0]["error"]

    @pytest.mark.asyncio
    async def test_bulk_scrape_single_url(self, async_client, base_url):
        """Bulk with a single URL behaves identically to single scrape."""
        resp = await async_client.post(
            "/api/scrape/bulk",
            json={"urls": [f"{base_url}/simple"], "scraper_type": "httpx"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_urls"] == 1
        assert data["completed"] == 1
        assert data["results"][0]["title"] == "Test Page"


# ─── Response Schema Contract ────────────────────────────────────────────────


class TestResponseContractE2E:
    """Verify response payloads match the documented schema exactly."""

    @pytest.mark.asyncio
    async def test_scrape_response_has_all_fields(self, async_client, base_url):
        """Every field in ScrapeResponse is present (not silently dropped)."""
        resp = await async_client.post(
            "/api/scrape/url",
            json={"url": f"{base_url}/simple", "scraper_type": "httpx"},
        )
        data = resp.json()
        expected_keys = {
            "url",
            "final_url",
            "html_content",
            "text_content",
            "title",
            "metadata",
            "links",
            "images",
            "screenshot_base64",
            "status",
            "error",
            "scraper_used",
            "processing_time_seconds",
        }
        assert expected_keys.issubset(data.keys())

    @pytest.mark.asyncio
    async def test_link_info_schema(self, async_client, base_url):
        """Each link in the response has text, url, rel, type."""
        resp = await async_client.post(
            "/api/scrape/url",
            json={"url": f"{base_url}/links", "scraper_type": "httpx"},
        )
        for link in resp.json()["links"]:
            assert "text" in link
            assert "url" in link
            assert "rel" in link
            assert "type" in link
            assert link["type"] in ("internal", "external")

    @pytest.mark.asyncio
    async def test_image_info_schema(self, async_client, base_url):
        """Each image has src, alt, title; content_base64 fields when downloaded."""
        resp = await async_client.post(
            "/api/scrape/url",
            json={
                "url": f"{base_url}/simple",
                "scraper_type": "httpx",
                "download_images": True,
            },
        )
        for img in resp.json()["images"]:
            assert "src" in img
            assert "alt" in img
            assert "title" in img

    @pytest.mark.asyncio
    async def test_bulk_response_has_all_fields(self, async_client, base_url):
        """BulkScrapeResponse has the expected top-level fields."""
        resp = await async_client.post(
            "/api/scrape/bulk",
            json={"urls": [f"{base_url}/simple"], "scraper_type": "httpx"},
        )
        data = resp.json()
        expected_keys = {
            "total_urls",
            "completed",
            "failed",
            "results",
            "processing_time_seconds",
        }
        assert expected_keys.issubset(data.keys())

    @pytest.mark.asyncio
    async def test_health_response_schema(self, async_client):
        """Health response has all documented fields."""
        resp = await async_client.get("/health")
        data = resp.json()
        for key in (
            "status",
            "timestamp",
            "service",
            "version",
            "playwright_available",
            "browser_ready",
        ):
            assert key in data, f"Missing key: {key}"
