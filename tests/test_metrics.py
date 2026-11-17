"""Tests for the /metrics endpoint and scrape/cache metric recording."""

from unittest.mock import AsyncMock, patch

import pytest

from services.base_scraper import ScrapeResult
from services.base_scraper import ScrapeStatus as SvcStatus


@pytest.mark.asyncio
async def test_metrics_endpoint_is_reachable_without_token(async_client):
    """/metrics is outside /api → auth-exempt, like /health*; returns exposition."""
    resp = await async_client.get("/metrics")
    assert resp.status_code == 200
    assert "text/plain" in resp.headers["content-type"]
    assert "version=" in resp.headers["content-type"]  # prometheus exposition
    # The metric families are registered (even if zero).
    assert "webscraper_scrapes_total" in resp.text
    assert "webscraper_cache_lookups_total" in resp.text


def test_scrape_increments_backend_counter():
    """record_scrape (called by the manager on every scrape) bumps the counter."""
    from services.metrics import SCRAPES_TOTAL, record_scrape

    before = SCRAPES_TOTAL.labels(backend="HttpxScraper", status="completed")._value.get()
    record_scrape("HttpxScraper", "completed", 0.4)
    after = SCRAPES_TOTAL.labels(backend="HttpxScraper", status="completed")._value.get()
    assert after == before + 1


class TestResultCacheRouting:
    """The single-URL endpoint serves a HIT from cache, bypasses on no_cache."""

    @staticmethod
    def _completed():
        return ScrapeResult(
            url="https://example.com",
            html_content="<html>ok</html>",
            text_content="ok",
            status=SvcStatus.COMPLETED,
            scraper_used="HttpxScraper",
            processing_time_seconds=0.2,
        )

    @pytest.mark.asyncio
    async def test_miss_then_hit_does_not_refetch(self, async_client, monkeypatch):
        from services.cache import get_result_cache

        get_result_cache().clear()
        mgr = AsyncMock()
        mgr.scrape_url = AsyncMock(return_value=self._completed())
        with (
            patch("routers.scrape.get_scraper_manager", return_value=mgr),
            patch("routers.scrape._guard_url", new=AsyncMock()),
        ):
            body = {"url": "https://example.com/cacheable", "scraper_type": "httpx"}
            r1 = await async_client.post("/api/scrape/url", json=body)
            r2 = await async_client.post("/api/scrape/url", json=body)
        assert r1.headers["X-Cache"] == "MISS"
        assert r2.headers["X-Cache"] == "HIT"
        # The HIT did not re-invoke the manager.
        assert mgr.scrape_url.call_count == 1
        assert r2.json()["text_content"] == "ok"

    @pytest.mark.asyncio
    async def test_no_cache_bypasses(self, async_client):
        from services.cache import get_result_cache

        get_result_cache().clear()
        mgr = AsyncMock()
        mgr.scrape_url = AsyncMock(return_value=self._completed())
        with (
            patch("routers.scrape.get_scraper_manager", return_value=mgr),
            patch("routers.scrape._guard_url", new=AsyncMock()),
        ):
            body = {"url": "https://example.com/x", "scraper_type": "httpx", "no_cache": True}
            r1 = await async_client.post("/api/scrape/url", json=body)
            r2 = await async_client.post("/api/scrape/url", json=body)
        assert r1.headers["X-Cache"] == "BYPASS"
        assert r2.headers["X-Cache"] == "BYPASS"
        assert mgr.scrape_url.call_count == 2  # always re-fetched

    @pytest.mark.asyncio
    async def test_failed_result_not_cached(self, async_client):
        from services.cache import get_result_cache

        get_result_cache().clear()
        failed = ScrapeResult(
            url="https://example.com/f",
            status=SvcStatus.FAILED,
            error="boom",
            scraper_used="HttpxScraper",
        )
        mgr = AsyncMock()
        mgr.scrape_url = AsyncMock(return_value=failed)
        with (
            patch("routers.scrape.get_scraper_manager", return_value=mgr),
            patch("routers.scrape._guard_url", new=AsyncMock()),
        ):
            body = {"url": "https://example.com/f", "scraper_type": "httpx"}
            r1 = await async_client.post("/api/scrape/url", json=body)
            r2 = await async_client.post("/api/scrape/url", json=body)
        assert r1.headers["X-Cache"] == "MISS"
        assert r2.headers["X-Cache"] == "MISS"  # FAILED never stored → still a miss
        assert mgr.scrape_url.call_count == 2
