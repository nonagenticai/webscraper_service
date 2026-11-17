"""Tests for scrape API endpoints.

Uses real ASGI transport to the FastAPI app and a local HTTP server
for scraping targets. External HTTP calls are served by the local server.
The auth middleware is bypassed so tests focus on scraping logic.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from routers.scrape import MAX_LINKS, _convert_links, _run_extraction
from schemas.scrape import BulkScrapeRequest, ExtractionSchema, ScrapeRequest


class TestScrapeRequestDefaults:
    """Default page-load wait condition on the public request model."""

    def test_wait_until_defaults_to_load(self):
        # Must be 'load', not 'networkidle': on live SPAs networkidle never settles,
        # so a networkidle default makes the router force it on every call and the
        # scrape returns empty content. networkidle is opt-in only.
        req = ScrapeRequest(url="https://example.com")
        assert req.wait_until == "load"

    def test_wait_until_networkidle_still_accepted(self):
        req = ScrapeRequest(url="https://example.com", wait_until="networkidle")
        assert req.wait_until == "networkidle"

    @pytest.mark.parametrize("value", ["load", "domcontentloaded", "networkidle", "commit"])
    def test_wait_until_all_playwright_values_accepted(self, value):
        req = ScrapeRequest(url="https://example.com", wait_until=value)
        assert req.wait_until == value

    def test_wait_until_invalid_value_rejected(self):
        # A bad value must fail validation up front (422), not slip through to
        # page.goto and return a FAILED 200.
        import pydantic

        with pytest.raises(pydantic.ValidationError):
            ScrapeRequest(url="https://example.com", wait_until="garbage")


class TestBulkScrapeRequestDefaults:
    """Per-page wait/delay options mirrored onto the bulk request model.

    These were honored on the single route but silently dropped on bulk, so a
    caller could not deepen SPA capture across a batch. Defaults preserve the
    prior behavior; an invalid wait_until must fail up front like the single route.
    """

    def test_wait_until_defaults_to_load(self):
        req = BulkScrapeRequest(urls=["https://example.com"])
        assert req.wait_until == "load"

    def test_delay_and_wait_for_selector_default_to_noop(self):
        req = BulkScrapeRequest(urls=["https://example.com"])
        assert req.delay == 0.0
        assert req.wait_for_selector is None

    def test_valid_wait_options_accepted(self):
        req = BulkScrapeRequest(
            urls=["https://example.com"],
            delay=2.5,
            wait_until="networkidle",
            wait_for_selector="#app",
        )
        assert req.delay == 2.5
        assert req.wait_until == "networkidle"
        assert req.wait_for_selector == "#app"

    def test_invalid_wait_until_value_rejected(self):
        import pydantic

        with pytest.raises(pydantic.ValidationError):
            BulkScrapeRequest(urls=["https://example.com"], wait_until="garbage")


class TestScrapeRequestEndpointValidation:
    """Request-level validation surfaced through the HTTP layer."""

    @pytest.mark.asyncio
    async def test_invalid_wait_until_returns_422(self, async_client):
        resp = await async_client.post(
            "/api/scrape/url",
            json={"url": "https://example.com", "wait_until": "garbage"},
        )
        assert resp.status_code == 422
        assert resp.json()["error"]["code"] == "VALIDATION_ERROR"


# ─── Single URL Scraping ─────────────────────────────────────────────────────


class TestScrapeUrlEndpoint:
    """Tests for POST /api/scrape/url endpoint."""

    @pytest.mark.asyncio
    async def test_scrape_url_success(self, async_client, base_url):
        """Test successful URL scraping with real HTTP pipeline."""
        resp = await async_client.post(
            "/api/scrape/url",
            json={"url": f"{base_url}/simple", "scraper_type": "httpx"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["url"] == f"{base_url}/simple"
        assert data["status"] == "completed"
        assert data["title"] == "Test Page"
        assert data["scraper_used"] == "HttpxScraper"
        assert "Hello World" in data["text_content"]
        assert "<h1>Hello World</h1>" in data["html_content"]
        assert len(data["links"]) == 2
        assert len(data["images"]) == 2
        assert data["processing_time_seconds"] > 0
        assert data["error"] is None

    @pytest.mark.asyncio
    async def test_scrape_url_with_options(self, async_client, base_url):
        """Test URL scraping with custom options - verifies they don't break scraping."""
        resp = await async_client.post(
            "/api/scrape/url",
            json={
                "url": f"{base_url}/simple",
                "timeout": 60,
                "viewport_width": 1280,
                "viewport_height": 720,
                "wait_until": "networkidle",
                "scraper_type": "httpx",
                "user_agent": "Custom Agent",
                "headers": {"X-Custom": "value"},
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        assert data["title"] == "Test Page"

    @pytest.mark.asyncio
    async def test_scrape_url_javascript_enabled_flows_to_config(self, async_client):
        """round-N+7: the request-level `javascript_enabled` flag must reach the
        ScrapeConfig handed to the manager (the AUTO selector honors it to force
        Playwright). Previously the field did not exist on the request schema."""
        from services.base_scraper import ScrapeResult
        from services.base_scraper import ScrapeStatus as SvcStatus

        result = ScrapeResult(
            url="https://example.com",
            html_content="<html>ok</html>",
            status=SvcStatus.COMPLETED,
            scraper_used="PlaywrightScraper",
        )
        with patch("routers.scrape.get_scraper_manager") as mock_get_manager:
            mock_manager = MagicMock()
            mock_manager.scrape_url = AsyncMock(return_value=result)
            mock_get_manager.return_value = mock_manager

            resp = await async_client.post(
                "/api/scrape/url",
                json={"url": "https://example.com", "javascript_enabled": True},
            )
        assert resp.status_code == 200
        passed_config = mock_manager.scrape_url.call_args.kwargs["config"]
        assert passed_config.javascript_enabled is True

    @pytest.mark.asyncio
    async def test_scrape_url_javascript_enabled_defaults_false(self, async_client):
        """Omitting the flag leaves the config default (False) — backward compatible."""
        from services.base_scraper import ScrapeResult
        from services.base_scraper import ScrapeStatus as SvcStatus

        result = ScrapeResult(
            url="https://example.com",
            html_content="<html>ok</html>",
            status=SvcStatus.COMPLETED,
            scraper_used="HttpxScraper",
        )
        with patch("routers.scrape.get_scraper_manager") as mock_get_manager:
            mock_manager = MagicMock()
            mock_manager.scrape_url = AsyncMock(return_value=result)
            mock_get_manager.return_value = mock_manager

            resp = await async_client.post(
                "/api/scrape/url",
                json={"url": "https://example.com", "scraper_type": "httpx"},
            )
        assert resp.status_code == 200
        assert mock_manager.scrape_url.call_args.kwargs["config"].javascript_enabled is False

    @pytest.mark.asyncio
    async def test_response_carries_markdown_and_structured_data(self, async_client):
        """The manager-derived markdown_content + structured_data are mapped onto
        the response (LLM/RAG-ready output). Additive + optional → backward compat."""
        from services.base_scraper import ScrapeResult
        from services.base_scraper import ScrapeStatus as SvcStatus

        result = ScrapeResult(
            url="https://example.com",
            html_content="<html>ok</html>",
            text_content="ok",
            markdown_content="# Heading\n\nBody text.",
            structured_data={"json-ld": [{"@type": "Article", "headline": "H"}]},
            status=SvcStatus.COMPLETED,
            scraper_used="HttpxScraper",
        )
        with patch("routers.scrape.get_scraper_manager") as mock_get_manager:
            mock_manager = MagicMock()
            mock_manager.scrape_url = AsyncMock(return_value=result)
            mock_get_manager.return_value = mock_manager

            resp = await async_client.post(
                "/api/scrape/url",
                json={"url": "https://example.com", "scraper_type": "httpx"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["markdown_content"] == "# Heading\n\nBody text."
        assert data["structured_data"]["json-ld"][0]["headline"] == "H"
        # Unchanged fields stay intact.
        assert data["text_content"] == "ok"

    @pytest.mark.asyncio
    async def test_response_defaults_when_extraction_absent(self, async_client):
        """A result without the new fields → markdown_content null, structured_data {}."""
        from services.base_scraper import ScrapeResult
        from services.base_scraper import ScrapeStatus as SvcStatus

        result = ScrapeResult(
            url="https://example.com",
            html_content="<html>ok</html>",
            text_content="ok",
            status=SvcStatus.COMPLETED,
            scraper_used="HttpxScraper",
        )
        with patch("routers.scrape.get_scraper_manager") as mock_get_manager:
            mock_manager = MagicMock()
            mock_manager.scrape_url = AsyncMock(return_value=result)
            mock_get_manager.return_value = mock_manager

            resp = await async_client.post(
                "/api/scrape/url",
                json={"url": "https://example.com", "scraper_type": "httpx"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["markdown_content"] is None
        assert data["structured_data"] == {}

    @pytest.mark.asyncio
    async def test_scrape_url_failed(self, async_client, base_url):
        """Test URL scraping that fails (server error)."""
        resp = await async_client.post(
            "/api/scrape/url",
            json={"url": f"{base_url}/error500", "scraper_type": "httpx"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "failed"
        assert "500" in data["error"]
        assert data["scraper_used"] == "HttpxScraper"

    @pytest.mark.asyncio
    async def test_scrape_url_exception(self, async_client):
        """Test URL scraping with unexpected exception from scraper manager."""
        with patch("routers.scrape.get_scraper_manager") as mock_get_manager:
            mock_manager = MagicMock()
            mock_manager.scrape_url = AsyncMock(side_effect=Exception("Unexpected error"))
            mock_get_manager.return_value = mock_manager

            resp = await async_client.post(
                "/api/scrape/url",
                json={"url": "https://example.com"},
            )
            assert resp.status_code == 500
            body = resp.json()
            assert "Scraping failed" in body["error"]["message"]
            assert "Unexpected error" in body["error"]["message"]

    @pytest.mark.asyncio
    async def test_scrape_url_oversized_content_clamped_not_500(self, async_client):
        """A page whose text/html exceeds the schema cap must be truncated to a
        clean 200, not raise a ValidationError → 500. Regression for the
        'scraped content must never 500' class."""
        from schemas.scrape import MAX_CONTENT_LENGTH
        from services.base_scraper import ScrapeResult
        from services.base_scraper import ScrapeStatus as SvcStatus

        oversized = "x" * (MAX_CONTENT_LENGTH + 5000)
        result = ScrapeResult(
            url="https://example.com",
            html_content=oversized,
            text_content=oversized,
            status=SvcStatus.COMPLETED,
            scraper_used="HttpxScraper",
        )
        with patch("routers.scrape.get_scraper_manager") as mock_get_manager:
            mock_manager = MagicMock()
            mock_manager.scrape_url = AsyncMock(return_value=result)
            mock_get_manager.return_value = mock_manager

            resp = await async_client.post(
                "/api/scrape/url",
                json={"url": "https://example.com", "scraper_type": "httpx"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        assert len(data["text_content"]) == MAX_CONTENT_LENGTH
        assert len(data["html_content"]) == MAX_CONTENT_LENGTH

    @pytest.mark.asyncio
    async def test_scrape_url_oversized_link_image_fields_clamped_not_500(self, async_client):
        """Per-link/per-image string fields exceeding their schema cap must be
        clamped to a clean 200, not raise ValidationError → 500. A long inline
        ``data:`` URI kept as an image src is the routine real-world trigger; the
        N+4 clamp covered only html/text/screenshot. Regression for the per-field
        gap in the 'scraped content must never 500' class."""
        from schemas.scrape import MAX_FIELD_LENGTH
        from services.base_scraper import ScrapeResult
        from services.base_scraper import ScrapeStatus as SvcStatus

        oversized_src = "data:image/png;base64," + ("A" * (MAX_FIELD_LENGTH + 5000))
        oversized_url = "https://example.com/" + ("a" * (MAX_FIELD_LENGTH + 5000))
        oversized_srcset = "https://example.com/x.jpg " + ("b" * (MAX_FIELD_LENGTH + 5000))
        result = ScrapeResult(
            url="https://example.com",
            html_content="<html></html>",
            text_content="ok",
            status=SvcStatus.COMPLETED,
            scraper_used="HttpxScraper",
            links=[{"text": "x" * (MAX_FIELD_LENGTH + 5000), "url": oversized_url, "type": None}],
            images=[{"src": oversized_src, "alt": "logo", "title": "", "srcset": oversized_srcset}],
        )
        with patch("routers.scrape.get_scraper_manager") as mock_get_manager:
            mock_manager = MagicMock()
            mock_manager.scrape_url = AsyncMock(return_value=result)
            mock_get_manager.return_value = mock_manager

            resp = await async_client.post(
                "/api/scrape/url",
                json={"url": "https://example.com", "scraper_type": "httpx"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        assert len(data["links"][0]["url"]) == MAX_FIELD_LENGTH
        assert len(data["links"][0]["text"]) == MAX_FIELD_LENGTH
        assert len(data["images"][0]["src"]) == MAX_FIELD_LENGTH
        assert len(data["images"][0]["srcset"]) == MAX_FIELD_LENGTH

    @pytest.mark.asyncio
    async def test_scrape_url_metadata_control_chars_sanitized(self, async_client):
        """metadata str values are boundary-cleaned like every other returned
        string (control chars from arbitrary <meta> must not reach consumers),
        while non-str values pass through untouched."""
        from services.base_scraper import ScrapeResult
        from services.base_scraper import ScrapeStatus as SvcStatus

        result = ScrapeResult(
            url="https://example.com",
            text_content="ok",
            status=SvcStatus.COMPLETED,
            scraper_used="HttpxScraper",
            metadata={"description": "hello\x07world", "status_code": 200},
        )
        with patch("routers.scrape.get_scraper_manager") as mock_get_manager:
            mock_manager = MagicMock()
            mock_manager.scrape_url = AsyncMock(return_value=result)
            mock_get_manager.return_value = mock_manager

            resp = await async_client.post(
                "/api/scrape/url",
                json={"url": "https://example.com", "scraper_type": "httpx"},
            )
        assert resp.status_code == 200
        meta = resp.json()["metadata"]
        assert "\x07" not in meta["description"]
        assert meta["description"] == "helloworld"
        assert meta["status_code"] == 200  # non-str value untouched

    @pytest.mark.asyncio
    async def test_scrape_url_invalid_request(self, async_client):
        """Test URL scraping with invalid request body."""
        resp = await async_client.post(
            "/api/scrape/url",
            json={"invalid": "data"},
        )
        assert resp.status_code == 422
        body = resp.json()
        assert body["error"]["code"] == "VALIDATION_ERROR"
        assert body["error"]["status"] == 422

    @pytest.mark.asyncio
    async def test_scrape_url_extracts_metadata(self, async_client, base_url):
        """Test that metadata (description, og tags, canonical) is extracted."""
        resp = await async_client.post(
            "/api/scrape/url",
            json={"url": f"{base_url}/simple", "scraper_type": "httpx"},
        )
        data = resp.json()
        meta = data["metadata"]
        assert meta["status_code"] == 200
        assert meta["description"] == "A simple test page"
        assert meta["og:title"] == "OG Test Page"
        assert "canonical_url" in meta

    @pytest.mark.asyncio
    async def test_scrape_url_classifies_links(self, async_client, base_url):
        """Test that links are classified as internal/external."""
        resp = await async_client.post(
            "/api/scrape/url",
            json={"url": f"{base_url}/simple", "scraper_type": "httpx"},
        )
        links = resp.json()["links"]
        internal = [lnk for lnk in links if lnk["type"] == "internal"]
        external = [lnk for lnk in links if lnk["type"] == "external"]
        assert len(internal) == 1
        assert internal[0]["text"] == "About"
        assert len(external) == 1
        assert "external.example.com" in external[0]["url"]

    @pytest.mark.asyncio
    async def test_scrape_url_nonexistent_returns_failed(self, async_client, base_url):
        """Test scraping a 404 page returns status=failed."""
        resp = await async_client.post(
            "/api/scrape/url",
            json={"url": f"{base_url}/no-such-page", "scraper_type": "httpx"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "failed"
        assert "404" in data["error"]


# ─── Bulk Scraping ───────────────────────────────────────────────────────────


class TestBulkScrapeEndpoint:
    """Tests for POST /api/scrape/bulk endpoint."""

    @pytest.mark.asyncio
    async def test_bulk_scrape_success(self, async_client, base_url):
        """Test successful bulk scraping with real HTTP pipeline."""
        resp = await async_client.post(
            "/api/scrape/bulk",
            json={
                "urls": [f"{base_url}/simple", f"{base_url}/links"],
                "scraper_type": "httpx",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_urls"] == 2
        assert data["completed"] == 2
        assert data["failed"] == 0
        assert len(data["results"]) == 2
        assert data["processing_time_seconds"] > 0

        titles = {r["title"] for r in data["results"]}
        assert "Test Page" in titles
        assert "Link Page" in titles

    @pytest.mark.asyncio
    async def test_bulk_scrape_partial_failure(self, async_client, base_url):
        """Test bulk scraping with some failures."""
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

        ok = [r for r in data["results"] if r["status"] == "completed"]
        fail = [r for r in data["results"] if r["status"] == "failed"]
        assert len(ok) == 1
        assert ok[0]["title"] == "Test Page"
        assert len(fail) == 1
        assert "500" in fail[0]["error"]

    @pytest.mark.asyncio
    async def test_bulk_oversized_field_does_not_fail_whole_batch(self, async_client):
        """One URL yielding an oversized per-image field must NOT 500 the entire
        /bulk batch — it is clamped and the other URLs still return."""
        from schemas.scrape import MAX_FIELD_LENGTH
        from services.base_scraper import ScrapeResult
        from services.base_scraper import ScrapeStatus as SvcStatus

        oversized_src = "data:image/png;base64," + ("A" * (MAX_FIELD_LENGTH + 5000))
        results = [
            ScrapeResult(
                url="https://big.example.com",
                text_content="big",
                status=SvcStatus.COMPLETED,
                scraper_used="HttpxScraper",
                images=[{"src": oversized_src, "alt": "", "title": ""}],
            ),
            ScrapeResult(
                url="https://clean.example.com",
                text_content="clean",
                title="Clean",
                status=SvcStatus.COMPLETED,
                scraper_used="HttpxScraper",
            ),
        ]
        with patch("routers.scrape.get_scraper_manager") as mock_get_manager:
            mock_manager = MagicMock()
            mock_manager.scrape_bulk = AsyncMock(return_value=results)
            mock_get_manager.return_value = mock_manager

            resp = await async_client.post(
                "/api/scrape/bulk",
                json={
                    "urls": ["https://big.example.com", "https://clean.example.com"],
                    "scraper_type": "httpx",
                },
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_urls"] == 2
        assert data["completed"] == 2
        by_url = {r["url"]: r for r in data["results"]}
        assert len(by_url["https://big.example.com"]["images"][0]["src"]) == MAX_FIELD_LENGTH
        assert by_url["https://clean.example.com"]["title"] == "Clean"

    @pytest.mark.asyncio
    async def test_bulk_scrape_with_options(self, async_client, base_url):
        """Test bulk scraping with options."""
        resp = await async_client.post(
            "/api/scrape/bulk",
            json={
                "urls": [f"{base_url}/simple"],
                "timeout": 60,
                "scraper_type": "httpx",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_urls"] == 1
        assert data["completed"] == 1
        assert data["results"][0]["title"] == "Test Page"

    @pytest.mark.asyncio
    async def test_bulk_scrape_exception(self, async_client):
        """Test bulk scraping with unexpected exception."""
        with patch("routers.scrape.get_scraper_manager") as mock_get_manager:
            mock_manager = MagicMock()
            mock_manager.scrape_bulk = AsyncMock(side_effect=Exception("Bulk error"))
            mock_get_manager.return_value = mock_manager

            resp = await async_client.post(
                "/api/scrape/bulk",
                json={"urls": ["https://example.com"]},
            )
            assert resp.status_code == 500
            assert "Bulk scraping failed" in resp.json()["error"]["message"]

    @pytest.mark.asyncio
    async def test_bulk_scrape_empty_results_metadata(self, async_client, base_url):
        """Test bulk scraping with empty page - None metadata handled gracefully."""
        resp = await async_client.post(
            "/api/scrape/bulk",
            json={"urls": [f"{base_url}/empty"], "scraper_type": "httpx"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["completed"] == 1
        assert data["results"][0]["title"] == "Empty"
        assert data["results"][0]["links"] == []
        assert data["results"][0]["images"] == []

    @pytest.mark.asyncio
    async def test_bulk_scrape_with_download_images_and_screenshot(self, async_client, base_url):
        """Test bulk scraping with download_images - local image gets base64."""
        resp = await async_client.post(
            "/api/scrape/bulk",
            json={
                "urls": [f"{base_url}/simple"],
                "scraper_type": "httpx",
                "download_images": True,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["completed"] == 1
        result = data["results"][0]
        # Find the local image (not the external one which will fail to download)
        local_img = next((i for i in result["images"] if "logo.png" in i["src"]), None)
        assert local_img is not None
        assert local_img["content_base64"] is not None
        assert len(local_img["content_base64"]) > 0


# ─── Scraper Type Conversion ────────────────────────────────────────────────


class TestScraperTypeConversion:
    """Tests for scraper type conversion functions."""

    @pytest.mark.asyncio
    async def test_convert_scraper_types(self, async_client, base_url):
        """Test that httpx scraper type works end-to-end."""
        resp = await async_client.post(
            "/api/scrape/url",
            json={"url": f"{base_url}/simple", "scraper_type": "httpx"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "completed"
        assert resp.json()["scraper_used"] == "HttpxScraper"

    @pytest.mark.asyncio
    async def test_invalid_scraper_type_rejected(self, async_client, base_url):
        """Test that an invalid scraper type is rejected with 422."""
        resp = await async_client.post(
            "/api/scrape/url",
            json={"url": f"{base_url}/simple", "scraper_type": "nonexistent"},
        )
        assert resp.status_code == 422


class TestScrapeSsrfGuard:
    """SSRF guard on the scrape endpoints (loopback is allowlisted in tests)."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "bad_url",
        [
            "http://169.254.169.254/latest/meta-data/",
            "http://10.0.0.1/",
            "http://registry.example.com/v2/",
            "http://localhost:9119/health",
        ],
    )
    async def test_single_blocked_url_returns_400(self, async_client, bad_url):
        resp = await async_client.post(
            "/api/scrape/url", json={"url": bad_url, "scraper_type": "httpx"}
        )
        assert resp.status_code == 400
        assert "not allowed" in resp.json()["error"]["message"].lower()

    @pytest.mark.asyncio
    async def test_blocked_login_url_returns_400(self, async_client, base_url):
        resp = await async_client.post(
            "/api/scrape/url",
            json={
                "url": f"{base_url}/simple",
                "scraper_type": "httpx",
                "auth_config": {
                    "type": "form",
                    "credential_id": "cred-x",
                    "login_url": "http://10.0.0.1/login",
                },
            },
        )
        assert resp.status_code == 400
        assert "login_url" in resp.json()["error"]["message"]

    @pytest.mark.asyncio
    async def test_bulk_blocked_url_is_per_url_failed(self, async_client, base_url):
        resp = await async_client.post(
            "/api/scrape/bulk",
            json={
                "urls": [f"{base_url}/simple", "http://169.254.169.254/"],
                "scraper_type": "httpx",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_urls"] == 2
        assert data["completed"] == 1
        assert data["failed"] == 1
        by_url = {r["url"]: r for r in data["results"]}
        assert by_url["http://169.254.169.254/"]["status"] == "failed"
        assert "not allowed" in by_url["http://169.254.169.254/"]["error"].lower()
        assert by_url[f"{base_url}/simple"]["status"] == "completed"


class TestConverterCountCaps:
    """Per-field lengths are clamped; the link/image COUNT is capped too so a
    page of pure <a>/<img> tags can't produce an unbounded response."""

    def test_links_capped_to_max(self):
        from routers.scrape import MAX_LINKS, _convert_links

        raw = [{"text": "t", "url": f"https://e.com/{i}"} for i in range(MAX_LINKS + 50)]
        links = _convert_links(raw)
        assert len(links) == MAX_LINKS

    def test_images_capped_to_max(self):
        from routers.scrape import MAX_IMAGES, _convert_images

        raw = [{"src": f"https://e.com/{i}.png", "alt": "a"} for i in range(MAX_IMAGES + 50)]
        images = _convert_images(raw)
        assert len(images) == MAX_IMAGES

    def test_under_cap_unchanged(self):
        from routers.scrape import _convert_images, _convert_links

        assert len(_convert_links([{"text": "t", "url": "https://e.com/x"}])) == 1
        assert len(_convert_images([{"src": "https://e.com/x.png", "alt": "a"}])) == 1


class TestLinkScoringContract:
    """WS1 additive-only guarantees for the links[] contract.

    ``w2v/analysis_service/src/domains/icp/routes.py`` unpacks each link as a
    dict of {text, url, rel, type} and reads ``links[].url``. That shape already
    broke once when it was assumed to be list[str], so scoring must ADD a field
    and never rename, reorder or drop one.
    """

    RAW = [
        {"url": "https://example.com/", "text": "Home", "rel": None, "type": "internal"},
        {"url": "https://example.com/pricing", "text": "Pricing", "rel": None, "type": "internal"},
        {"url": "https://x.com/acme", "text": "Follow", "rel": "nofollow", "type": "external"},
    ]

    def test_existing_keys_are_preserved_exactly(self):
        converted = _convert_links(self.RAW)
        assert [link.model_dump() for link in converted] == [
            {**raw, "score": link.score} for raw, link in zip(self.RAW, converted, strict=True)
        ]

    def test_document_order_is_the_default(self):
        assert [link.url for link in _convert_links(self.RAW)] == [r["url"] for r in self.RAW]

    def test_every_link_carries_a_bounded_score_without_a_query(self):
        for link in _convert_links(self.RAW):
            assert 0.0 <= link.score <= 1.0

    def test_sorting_is_opt_in(self):
        ranked = _convert_links(self.RAW, "pricing", sort_by_score=True)
        assert ranked[0].url == "https://example.com/pricing"
        assert [link.url for link in _convert_links(self.RAW, "pricing")] == [
            r["url"] for r in self.RAW
        ]

    def test_scoring_runs_before_the_max_links_cap(self):
        """The best link must survive truncation even when it is last on the page.

        Capping first and scoring second would silently discard exactly the link
        the caller asked for on any page with more than MAX_LINKS anchors.
        """
        filler = [
            {"url": f"https://example.com/f/{i}", "text": "x", "rel": None, "type": "internal"}
            for i in range(MAX_LINKS + 50)
        ]
        target = {
            "url": "https://example.com/pricing",
            "text": "Pricing",
            "rel": None,
            "type": "internal",
        }
        converted = _convert_links([*filler, target], "pricing", sort_by_score=True)
        assert len(converted) == MAX_LINKS
        assert converted[0].url == target["url"]


class TestExtractionRouterWiring:
    """`extracted_data` must distinguish 'not asked for' from 'matched nothing'."""

    HTML = '<html><body><li class="row"><b>A</b></li><li class="row"><b>B</b></li></body></html>'

    def _schema(self, base="li.row"):
        return ExtractionSchema.model_validate(
            {"baseSelector": base, "fields": [{"name": "v", "selector": "b", "type": "text"}]}
        )

    def test_none_when_no_schema_supplied(self):
        assert _run_extraction(self.HTML, None) == (None, False, [])

    def test_empty_list_when_a_schema_was_supplied_but_there_is_no_html(self):
        """None means "you didn't ask"; it must not also mean "no HTML".

        A backend that returned nothing used to collapse into the same `null`
        as a request carrying no schema at all, so a caller could not tell a
        failed page from one they had never asked to extract from.
        """
        assert _run_extraction(None, self._schema()) == ([], False, [])
        assert _run_extraction("", self._schema()) == ([], False, [])

    def test_empty_list_when_schema_ran_but_matched_nothing(self):
        assert _run_extraction(self.HTML, self._schema("table.absent")) == ([], False, [])

    def test_rows_when_schema_matches(self):
        assert _run_extraction(self.HTML, self._schema()) == (
            [{"v": "A"}, {"v": "B"}],
            False,
            [],
        )


class TestNewOptionsReachTheResponse:
    """End-to-end wiring for the PR #49 options.

    Every one of these was previously unasserted at the endpoint: deleting the
    plumbing between ScrapeRequest and the response kept the suite green, which
    is precisely how a silently-ignored option ships.
    """

    @staticmethod
    def _result(html: str):
        from services.base_scraper import ScrapeResult
        from services.base_scraper import ScrapeStatus as SvcStatus

        return ScrapeResult(
            url="https://example.com",
            html_content=html,
            text_content="ok",
            status=SvcStatus.COMPLETED,
            scraper_used="HttpxScraper",
        )

    @pytest.mark.asyncio
    async def test_extracted_data_reaches_the_json_body(self, async_client):
        html = (
            '<html><body><li class="row"><b>A</b></li><li class="row"><b>B</b></li></body></html>'
        )
        with patch("routers.scrape.get_scraper_manager") as mock_get_manager:
            mock_manager = MagicMock()
            mock_manager.scrape_url = AsyncMock(return_value=self._result(html))
            mock_get_manager.return_value = mock_manager

            resp = await async_client.post(
                "/api/scrape/url",
                json={
                    "url": "https://example.com",
                    "scraper_type": "httpx",
                    "extraction_schema": {
                        "baseSelector": "li.row",
                        "fields": [{"name": "v", "selector": "b", "type": "text"}],
                    },
                },
            )
        assert resp.status_code == 200
        assert resp.json()["extracted_data"] == [{"v": "A"}, {"v": "B"}]

    @pytest.mark.asyncio
    async def test_extracted_data_is_null_when_no_schema_was_sent(self, async_client):
        with patch("routers.scrape.get_scraper_manager") as mock_get_manager:
            mock_manager = MagicMock()
            mock_manager.scrape_url = AsyncMock(return_value=self._result("<html>ok</html>"))
            mock_get_manager.return_value = mock_manager

            resp = await async_client.post(
                "/api/scrape/url",
                json={"url": "https://example.com", "scraper_type": "httpx"},
            )
        assert resp.status_code == 200
        assert resp.json()["extracted_data"] is None

    @pytest.mark.asyncio
    async def test_virtual_scroll_reaches_the_scraper_config(self, async_client):
        with patch("routers.scrape.get_scraper_manager") as mock_get_manager:
            mock_manager = MagicMock()
            mock_manager.scrape_url = AsyncMock(return_value=self._result("<html>ok</html>"))
            mock_get_manager.return_value = mock_manager

            resp = await async_client.post(
                "/api/scrape/url",
                json={
                    "url": "https://example.com",
                    "virtual_scroll": {"container_selector": "#feed", "scroll_count": 4},
                    "stealth": True,
                },
            )
        assert resp.status_code == 200
        cfg = mock_manager.scrape_url.call_args.kwargs["config"]
        assert cfg.virtual_scroll is not None
        assert cfg.virtual_scroll.container_selector == "#feed"
        assert cfg.stealth is True

    @pytest.mark.asyncio
    async def test_browser_only_options_are_rejected_for_a_non_browser_backend(self, async_client):
        """422, not a first-viewport result reported as completed."""
        resp = await async_client.post(
            "/api/scrape/url",
            json={
                "url": "https://example.com",
                "scraper_type": "httpx",
                "virtual_scroll": {"container_selector": "#feed"},
            },
        )
        assert resp.status_code == 422
        assert "JavaScript-capable backend" in resp.text

    @pytest.mark.asyncio
    async def test_stealth_is_rejected_for_a_non_browser_backend(self, async_client):
        resp = await async_client.post(
            "/api/scrape/url",
            json={"url": "https://example.com", "scraper_type": "curl_cffi", "stealth": True},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_invalid_extraction_selector_is_422_not_an_empty_result(self, async_client):
        resp = await async_client.post(
            "/api/scrape/url",
            json={
                "url": "https://example.com",
                "extraction_schema": {
                    "baseSelector": "div:::broken",
                    "fields": [{"name": "v", "type": "text"}],
                },
            },
        )
        assert resp.status_code == 422
        assert "not a valid CSS selector" in resp.text

    @pytest.mark.asyncio
    async def test_bulk_rejects_browser_options_combined_with_crawling(self, async_client):
        """follow_links routes the batch to Scrapy, which cannot scroll."""
        resp = await async_client.post(
            "/api/scrape/bulk",
            json={
                "urls": ["https://example.com"],
                "follow_links": True,
                "virtual_scroll": {"container_selector": "#feed"},
            },
        )
        assert resp.status_code == 422
        assert "follow_links" in resp.text

    @pytest.mark.asyncio
    async def test_bulk_allowed_domains_reaches_the_crawl_config(self, async_client):
        """It was accepted and documented, and then never consulted."""
        with patch("routers.scrape.get_scraper_manager") as mock_get_manager:
            mock_manager = MagicMock()
            mock_manager.scrape_bulk = AsyncMock(return_value=[self._result("<html>ok</html>")])
            mock_get_manager.return_value = mock_manager

            resp = await async_client.post(
                "/api/scrape/bulk",
                json={
                    "urls": ["https://example.com"],
                    "follow_links": True,
                    "allowed_domains": ["example.com"],
                },
            )
        assert resp.status_code == 200
        cfg = mock_manager.scrape_bulk.call_args.kwargs["config"]
        assert cfg.allowed_domains == ["example.com"]

    @pytest.mark.asyncio
    async def test_bulk_wait_options_reach_the_scraper(self, async_client):
        """delay flows via ScrapeConfig; wait_until/wait_for_selector as kwargs.

        All three existed on the single route but were dropped on bulk, so a
        caller could not deepen SPA capture across a batch. This asserts the
        handler now plumbs them through to scraper_manager.scrape_bulk."""
        with patch("routers.scrape.get_scraper_manager") as mock_get_manager:
            mock_manager = MagicMock()
            mock_manager.scrape_bulk = AsyncMock(return_value=[self._result("<html>ok</html>")])
            mock_get_manager.return_value = mock_manager

            resp = await async_client.post(
                "/api/scrape/bulk",
                json={
                    "urls": ["https://example.com"],
                    "scraper_type": "playwright",
                    "delay": 3.0,
                    "wait_until": "networkidle",
                    "wait_for_selector": "#app",
                },
            )
        assert resp.status_code == 200
        kwargs = mock_manager.scrape_bulk.call_args.kwargs
        assert kwargs["config"].delay == 3.0
        assert kwargs["wait_until"] == "networkidle"
        assert kwargs["wait_for_selector"] == "#app"


class TestAllowedDomainsValidation:
    """A scope filter that silently matches nothing is worse than none."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "entry", ["https://example.com/", "*.example.com", "example.com:8080", " example.com"]
    )
    async def test_unmatchable_allowed_domains_entries_are_rejected(self, async_client, entry):
        """The filter compares against a bare host, so each of these — all
        natural guesses — matched no link at all and produced an empty crawl
        reported as a successful one."""
        resp = await async_client.post(
            "/api/scrape/bulk",
            json={
                "urls": ["https://example.com"],
                "follow_links": True,
                "allowed_domains": [entry],
            },
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_bare_hostnames_are_accepted(self, async_client):
        from services.base_scraper import ScrapeResult
        from services.base_scraper import ScrapeStatus as SvcStatus

        result = ScrapeResult(
            url="https://example.com",
            html_content="<html>ok</html>",
            status=SvcStatus.COMPLETED,
            scraper_used="ScrapyScraper",
        )
        with patch("routers.scrape.get_scraper_manager") as mock_get_manager:
            mock_manager = MagicMock()
            mock_manager.scrape_bulk = AsyncMock(return_value=[result])
            mock_get_manager.return_value = mock_manager
            resp = await async_client.post(
                "/api/scrape/bulk",
                json={
                    "urls": ["https://example.com"],
                    "follow_links": True,
                    "allowed_domains": ["example.com", "cdn.example.co.uk"],
                },
            )
        assert resp.status_code == 200
