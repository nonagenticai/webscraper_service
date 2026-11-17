"""Tests for main.py coverage: CORS config, lifespan, custom OpenAPI, root endpoint.

Uses real ASGI transport where possible. Only patches environment or specific
internal methods to test configuration paths.
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestGetCorsConfig:
    def test_default_origins(self):
        """Without CORS_ALLOW_ORIGINS env, returns defaults with credentials."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CORS_ALLOW_ORIGINS", None)
            from main import get_cors_config

            origins, allow_credentials = get_cors_config()
            assert "http://localhost:8921" in origins
            assert "http://localhost:3000" in origins
            assert allow_credentials is True

    def test_custom_origins(self):
        """CORS_ALLOW_ORIGINS env sets custom origins."""
        with patch.dict(
            os.environ, {"CORS_ALLOW_ORIGINS": "https://app.example.com, https://other.com"}
        ):
            from main import get_cors_config

            origins, allow_credentials = get_cors_config()
            assert "https://app.example.com" in origins
            assert "https://other.com" in origins
            assert allow_credentials is True

    def test_wildcard_origin(self):
        """Wildcard origin disables credentials."""
        with patch.dict(os.environ, {"CORS_ALLOW_ORIGINS": "*"}):
            from main import get_cors_config

            origins, allow_credentials = get_cors_config()
            assert origins == ["*"]
            assert allow_credentials is False

    def test_wildcard_with_others(self):
        """Wildcard mixed with other origins still results in ['*'] and no credentials."""
        with patch.dict(os.environ, {"CORS_ALLOW_ORIGINS": "https://app.com,*,https://other.com"}):
            from main import get_cors_config

            origins, allow_credentials = get_cors_config()
            assert origins == ["*"]
            assert allow_credentials is False

    def test_empty_origins_string(self):
        """Empty string falls back to defaults."""
        with patch.dict(os.environ, {"CORS_ALLOW_ORIGINS": ""}):
            from main import get_cors_config

            origins, allow_credentials = get_cors_config()
            assert len(origins) > 0
            assert allow_credentials is True


class TestCustomOpenAPI:
    def test_custom_openapi_schema_generated(self):
        """custom_openapi produces a schema with BearerAuth and TenantIdHeader."""
        from main import app

        app.openapi_schema = None
        schema = app.openapi()
        assert "components" in schema
        assert "securitySchemes" in schema["components"]
        assert "BearerAuth" in schema["components"]["securitySchemes"]
        assert schema["components"]["securitySchemes"]["BearerAuth"]["type"] == "http"
        assert schema["components"]["securitySchemes"]["BearerAuth"]["scheme"] == "bearer"

        assert "parameters" in schema["components"]
        assert "TenantIdHeader" in schema["components"]["parameters"]
        tenant_param = schema["components"]["parameters"]["TenantIdHeader"]
        assert tenant_param["name"] == "X-Tenant-Id"
        assert tenant_param["in"] == "header"

    def test_custom_openapi_caches(self):
        """Second call returns cached schema."""
        from main import app

        app.openapi_schema = None
        schema1 = app.openapi()
        schema2 = app.openapi()
        assert schema1 is schema2

    def test_custom_openapi_adds_tenant_param_to_operations(self):
        """Each operation gets a TenantIdHeader $ref parameter."""
        from main import app

        app.openapi_schema = None
        schema = app.openapi()

        for path, path_item in schema.get("paths", {}).items():
            for method, operation in path_item.items():
                if not isinstance(operation, dict):
                    continue
                params = operation.get("parameters", [])
                tenant_refs = [
                    p
                    for p in params
                    if isinstance(p, dict)
                    and p.get("$ref") == "#/components/parameters/TenantIdHeader"
                ]
                assert len(tenant_refs) >= 1, f"Missing TenantIdHeader on {method.upper()} {path}"


class TestLifespan:
    @pytest.mark.asyncio
    async def test_lifespan_startup_shutdown(self):
        """Lifespan context manager initializes and closes scraper manager."""
        from main import app, lifespan

        mock_manager = MagicMock()
        mock_manager.close = AsyncMock()

        with patch("main.get_scraper_manager", return_value=mock_manager):
            async with lifespan(app):
                pass
            mock_manager.close.assert_awaited_once()


class TestRootEndpoint:
    @pytest.mark.asyncio
    async def test_root_returns_service_info(self, async_client):
        """Root endpoint returns service info with all documented fields."""
        resp = await async_client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "online"
        assert data["service"] == "webscraper-service"
        assert "version" in data
        assert data["endpoints"]["health"] == "/health"
        assert data["endpoints"]["docs"] == "/docs"
        assert data["endpoints"]["scrape_url"] == "/api/scrape/url"
        assert data["endpoints"]["scrape_bulk"] == "/api/scrape/bulk"
