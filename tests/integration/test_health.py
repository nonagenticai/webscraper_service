"""Integration tests for health check endpoints."""

import pytest

pytestmark = pytest.mark.integration


class TestHealthEndpoints:
    """Test health check endpoints."""

    def test_health_check(self, scraper_client):
        """Test GET /health endpoint."""
        response = scraper_client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] in ["healthy", "unhealthy"]
        assert "timestamp" in data
        assert "service" in data
        assert "version" in data
        assert "playwright_available" in data
        assert "browser_ready" in data

    def test_readiness_probe(self, scraper_client):
        """Test GET /health/ready endpoint."""
        response = scraper_client.get("/health/ready")
        # Accept both 200 (ready) and 503 (not ready)
        assert response.status_code in [200, 503]

        data = response.json()
        assert data["status"] in ["ready", "not_ready"]
        assert "timestamp" in data
        assert "checks" in data
        assert "playwright_available" in data["checks"]
        assert "browser_ready" in data["checks"]

    def test_liveness_probe(self, scraper_client):
        """Test GET /health/live endpoint."""
        response = scraper_client.get("/health/live")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "alive"
        assert "timestamp" in data
        assert "service" in data
