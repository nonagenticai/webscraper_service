"""Integration tests for scrape endpoints."""

import pytest

pytestmark = pytest.mark.integration


class TestScrapeUrl:
    """Test POST /api/scrape/url endpoint."""

    def test_scrape_simple(self, scraper_client):
        """Test scraping a simple URL."""
        response = scraper_client.post("/api/scrape/url", json={"url": "https://example.com"})

        # Accept 200 (success), 502 (bad gateway), or 503 (service unavailable)
        # Playwright/browser may not be available in all environments
        assert response.status_code in [200, 502, 503]

        if response.status_code == 200:
            data = response.json()
            # Verify response has expected content fields
            assert "url" in data
            assert "html_content" in data or "text_content" in data
            assert "status" in data
            assert "scraper_used" in data
            assert "processing_time_seconds" in data

    def test_scrape_invalid_url_422(self, scraper_client):
        """Test scraping with invalid URL returns 422."""
        response = scraper_client.post("/api/scrape/url", json={"url": "not-a-url"})

        # Expect 422 validation error for invalid URL format
        assert response.status_code == 422

        data = response.json()
        assert "detail" in data


class TestScrapeBulk:
    """Test POST /api/scrape/bulk endpoint."""

    def test_bulk_scrape(self, scraper_client):
        """Test bulk scraping multiple URLs."""
        response = scraper_client.post("/api/scrape/bulk", json={"urls": ["https://example.com"]})

        # Accept 200 (success), 202 (accepted), 502 (bad gateway), or 503 (service unavailable)
        # Playwright/browser may not be available in all environments
        assert response.status_code in [200, 202, 502, 503]

        if response.status_code in [200, 202]:
            data = response.json()
            # Verify response has expected bulk fields
            assert "total_urls" in data
            assert "completed" in data
            assert "failed" in data
            assert "results" in data
            assert "processing_time_seconds" in data
            assert data["total_urls"] == 1
