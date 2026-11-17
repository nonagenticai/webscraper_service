"""Tests for health.py edge cases: readiness probe exception/not-ready, health check exception.

Uses real ASGI transport. Patches only the specific internal dependencies
to trigger error paths.
"""

from unittest.mock import MagicMock, patch

import pytest


class TestReadinessProbeNotReady:
    @pytest.mark.asyncio
    async def test_readiness_not_ready_when_playwright_unavailable(self, async_client):
        """Readiness returns 503 when playwright is not available."""
        mock_manager = MagicMock()
        mock_manager.is_browser_ready.return_value = False

        with (
            patch("routers.health.get_scraper_manager", return_value=mock_manager),
            patch("importlib.util.find_spec", return_value=None),
        ):
            resp = await async_client.get("/health/ready")

        assert resp.status_code == 503
        data = resp.json()
        assert data["status"] == "not_ready"
        assert data["checks"]["playwright_available"] is False
        assert data["checks"]["browser_ready"] is False

    @pytest.mark.asyncio
    async def test_readiness_exception_returns_503(self, async_client):
        """Readiness returns 503 with error field when exception occurs."""
        with patch(
            "routers.health.get_scraper_manager",
            side_effect=RuntimeError("Manager init failed"),
        ):
            resp = await async_client.get("/health/ready")

        assert resp.status_code == 503
        data = resp.json()
        assert data["status"] == "not_ready"
        assert "Manager init failed" in data["error"]

    @pytest.mark.asyncio
    async def test_readiness_import_error(self, async_client):
        """Readiness handles ImportError for playwright gracefully."""
        mock_manager = MagicMock()
        mock_manager.is_browser_ready.return_value = False

        with (
            patch("routers.health.get_scraper_manager", return_value=mock_manager),
            patch("importlib.util.find_spec", side_effect=ImportError("No module")),
        ):
            resp = await async_client.get("/health/ready")

        assert resp.status_code in (200, 503)
        data = resp.json()
        assert data["checks"]["playwright_available"] is False


class TestHealthCheckException:
    @pytest.mark.asyncio
    async def test_health_check_exception_returns_unhealthy(self, async_client):
        """Health check returns unhealthy when exception is raised."""
        with patch(
            "routers.health.get_scraper_manager",
            side_effect=RuntimeError("Unexpected error"),
        ):
            resp = await async_client.get("/health")

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "unhealthy"
        assert data["playwright_available"] is False
        assert data["browser_ready"] is False
        assert data["service"] == "webscraper-service"


class TestReadinessReflectsTheBrowser:
    """Readiness must be DERIVED from every check, not just the cheapest one."""

    @pytest.mark.asyncio
    async def test_dead_browser_is_not_ready(self, async_client):
        """Regression: /health/ready was effectively a static 200.

        It gated only on playwright_available — which is true by construction
        in this image — so a pod whose Chromium had died still advertised
        itself as ready, kept its place in the Service, and answered every JS
        page with a first-viewport result. browser_ready was computed and
        thrown away.
        """
        mock_manager = MagicMock()
        mock_manager.is_browser_ready.return_value = False

        with (
            patch("routers.health.get_scraper_manager", return_value=mock_manager),
            patch("importlib.util.find_spec", return_value=object()),
        ):
            resp = await async_client.get("/health/ready")

        assert resp.status_code == 503
        data = resp.json()
        assert data["status"] == "not_ready"
        assert data["checks"]["playwright_available"] is True
        assert data["checks"]["browser_ready"] is False

    @pytest.mark.asyncio
    async def test_healthy_browser_is_ready(self, async_client):
        mock_manager = MagicMock()
        mock_manager.is_browser_ready.return_value = True

        with (
            patch("routers.health.get_scraper_manager", return_value=mock_manager),
            patch("importlib.util.find_spec", return_value=object()),
        ):
            resp = await async_client.get("/health/ready")

        assert resp.status_code == 200
        assert resp.json()["status"] == "ready"

    @pytest.mark.asyncio
    async def test_health_summary_does_not_claim_healthy_with_a_dead_browser(self, async_client):
        """A summary that says "healthy" next to browser_ready=false contradicts
        itself, and anything alerting on `status` never saw the outage."""
        mock_manager = MagicMock()
        mock_manager.is_browser_ready.return_value = False

        with (
            patch("routers.health.get_scraper_manager", return_value=mock_manager),
            patch("importlib.util.find_spec", return_value=object()),
        ):
            resp = await async_client.get("/health")

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "degraded"
        assert data["browser_ready"] is False
