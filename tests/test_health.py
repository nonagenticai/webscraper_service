"""Tests for health endpoints.

Uses real ASGI transport to the FastAPI app.
"""

import pytest


@pytest.mark.asyncio
async def test_root_endpoint(async_client):
    """Test root endpoint returns service info with correct structure."""
    resp = await async_client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["service"] == "webscraper-service"
    assert data["status"] == "online"
    assert "endpoints" in data
    assert data["endpoints"]["health"] == "/health"
    assert data["endpoints"]["scrape_url"] == "/api/scrape/url"
    assert data["endpoints"]["scrape_bulk"] == "/api/scrape/bulk"
    assert data["endpoints"]["docs"] == "/docs"


@pytest.mark.asyncio
async def test_health_endpoint(async_client):
    """Test health endpoint returns full health response."""
    resp = await async_client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in ("healthy", "unhealthy")
    assert "timestamp" in data
    assert data["service"] == "webscraper-service"
    assert data["version"] == "1.0.0"
    assert "playwright_available" in data
    assert "browser_ready" in data


@pytest.mark.asyncio
async def test_liveness_probe(async_client):
    """Test liveness probe returns alive status with timestamp."""
    resp = await async_client.get("/health/live")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "alive"
    assert "timestamp" in data
    assert data["service"] == "webscraper-service"


@pytest.mark.asyncio
async def test_readiness_probe(async_client):
    """Test readiness probe returns status with checks."""
    resp = await async_client.get("/health/ready")
    assert resp.status_code in (200, 503)
    data = resp.json()
    assert data["status"] in ("ready", "not_ready")
    assert "checks" in data
    assert "playwright_available" in data["checks"]
    assert "browser_ready" in data["checks"]
    assert "timestamp" in data
