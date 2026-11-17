"""Tests for license-based authentication middleware."""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from auth.middleware import (
    PUBLIC_PATHS,
    AuthMiddleware,
    LicenseServiceUnavailableError,
    _introspect_license,
    _is_trusted_network,
    _license_cache,
)


class TestAuthMiddleware:
    """Tests for AuthMiddleware."""

    def test_public_paths_constant(self):
        """Test that PUBLIC_PATHS contains expected paths."""
        assert "/" in PUBLIC_PATHS
        assert "/health" in PUBLIC_PATHS
        assert "/health/ready" in PUBLIC_PATHS
        assert "/health/live" in PUBLIC_PATHS
        assert "/docs" in PUBLIC_PATHS
        assert "/openapi.json" in PUBLIC_PATHS

    def test_middleware_init(self):
        """Test middleware initialization."""
        mock_app = MagicMock()
        middleware = AuthMiddleware(
            mock_app,
            protected_prefixes=["/api"],
        )
        assert middleware.protected_prefixes == ["/api"]

    def test_middleware_init_defaults(self):
        """Test middleware initialization with defaults."""
        mock_app = MagicMock()
        middleware = AuthMiddleware(mock_app)
        assert middleware.protected_prefixes == []
        assert middleware.bypass_auth is False

    def test_middleware_init_bypass(self):
        """Test middleware initialization with bypass_auth=True."""
        mock_app = MagicMock()
        middleware = AuthMiddleware(mock_app, bypass_auth=True)
        assert middleware.bypass_auth is True

    def test_is_public_path(self):
        """Test _is_public_path method."""
        mock_app = MagicMock()
        middleware = AuthMiddleware(mock_app)

        assert middleware._is_public_path("/") is True
        assert middleware._is_public_path("/health") is True
        assert middleware._is_public_path("/health/") is True
        assert middleware._is_public_path("/docs") is True
        assert middleware._is_public_path("/docs/something") is True
        assert middleware._is_public_path("/api/scrape") is False

    def test_is_public_path_static_prefix(self):
        """Test _is_public_path with /static prefix."""
        mock_app = MagicMock()
        middleware = AuthMiddleware(mock_app)
        assert middleware._is_public_path("/static/css/style.css") is True

    def test_is_protected_path(self):
        """Test _is_protected_path method."""
        mock_app = MagicMock()
        middleware = AuthMiddleware(mock_app, protected_prefixes=["/api"])

        assert middleware._is_protected_path("/api") is True
        assert middleware._is_protected_path("/api/scrape") is True
        assert middleware._is_protected_path("/api/") is True
        assert middleware._is_protected_path("/other") is False
        assert middleware._is_protected_path("/") is False

    def test_extract_license_token(self):
        """Test _extract_license_token method."""
        mock_app = MagicMock()
        middleware = AuthMiddleware(mock_app)

        # Mock request with X-Tenant-Id header
        mock_request = MagicMock()
        mock_request.headers.get.return_value = "Bearer test-token-123"
        token = middleware._extract_license_token(mock_request)
        assert token == "test-token-123"

        # Mock request without Bearer prefix
        mock_request.headers.get.return_value = "plain-token"
        token = middleware._extract_license_token(mock_request)
        assert token is None

        # Mock request with no header
        mock_request.headers.get.return_value = None
        token = middleware._extract_license_token(mock_request)
        assert token is None

    def test_extract_license_token_case_insensitive(self):
        """Test _extract_license_token with different Bearer casings."""
        mock_app = MagicMock()
        middleware = AuthMiddleware(mock_app)
        mock_request = MagicMock()

        mock_request.headers.get.return_value = "bearer my-token"
        token = middleware._extract_license_token(mock_request)
        assert token == "my-token"

        mock_request.headers.get.return_value = "BEARER MY-TOKEN"
        token = middleware._extract_license_token(mock_request)
        assert token == "MY-TOKEN"

    def test_unauthorized_response(self):
        """Test _unauthorized_response returns correct JSON structure."""
        mock_app = MagicMock()
        middleware = AuthMiddleware(mock_app)
        resp = middleware._unauthorized_response("Test error")
        assert resp.status_code == 401
        assert resp.headers.get("www-authenticate") == "Bearer"

    def test_internal_error_response(self):
        """Test _internal_error_response returns correct JSON structure."""
        mock_app = MagicMock()
        middleware = AuthMiddleware(mock_app)
        resp = middleware._internal_error_response("Internal error")
        assert resp.status_code == 500


class TestIsTrustedNetwork:
    """Tests for _is_trusted_network function."""

    def test_none_host_returns_false(self):
        """None host is not trusted."""
        assert _is_trusted_network(None) is False

    def test_empty_host_returns_false(self):
        """Empty host is not trusted."""
        assert _is_trusted_network("") is False

    def test_loopback_is_trusted(self):
        """127.0.0.1 is trusted."""
        assert _is_trusted_network("127.0.0.1") is True

    def test_ipv6_loopback_is_trusted(self):
        """::1 is trusted."""
        assert _is_trusted_network("::1") is True

    def test_private_network_is_trusted(self):
        """192.168.x.x is trusted."""
        assert _is_trusted_network("192.168.1.100") is True

    def test_10_network_is_trusted(self):
        """10.x.x.x is trusted."""
        assert _is_trusted_network("10.42.0.5") is True

    def test_172_16_network_is_trusted(self):
        """172.16.x.x is trusted."""
        assert _is_trusted_network("172.16.5.10") is True

    def test_public_ip_not_trusted(self):
        """Public IP is not trusted."""
        assert _is_trusted_network("8.8.8.8") is False

    def test_invalid_ip_returns_false(self):
        """Invalid IP string returns False."""
        assert _is_trusted_network("not-an-ip") is False

    def test_hostname_returns_false(self):
        """Hostname (not IP) returns False."""
        assert _is_trusted_network("example.com") is False


class TestIntrospectLicense:
    """Tests for _introspect_license function."""

    @pytest.fixture(autouse=True)
    def clear_cache(self):
        """Clear license cache before each test."""
        _license_cache.clear()
        yield
        _license_cache.clear()

    @pytest.mark.asyncio
    async def test_introspect_cache_hit(self):
        """Cached license result is returned without HTTP call."""
        import hashlib

        token = "cached-token"
        cache_key = f"lic:{hashlib.sha256(token.encode()).hexdigest()[:16]}"
        cached_data = {"active": True, "subject_id": "cached-tenant"}
        # Positive cache stores (data, expires_at); seed a future expiry.
        _license_cache[cache_key] = (cached_data, time.time() + 300)

        result = await _introspect_license(token)
        assert result == cached_data
        assert result["subject_id"] == "cached-tenant"

    @pytest.mark.asyncio
    async def test_introspect_success_active(self):
        """Successful introspection with active=True caches and returns data."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"active": True, "subject_id": "test-tenant"}

        with patch("auth.middleware.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await _introspect_license("my-token")

        assert result is not None
        assert result["active"] is True
        assert result["subject_id"] == "test-tenant"
        # Should be cached now
        assert len(_license_cache) == 1

    @pytest.mark.asyncio
    async def test_introspect_success_inactive(self):
        """Successful introspection with active=False returns data but does not cache."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"active": False}

        with patch("auth.middleware.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await _introspect_license("inactive-token")

        assert result is not None
        assert result["active"] is False
        # Should NOT be cached (inactive)
        assert len(_license_cache) == 0

    @pytest.mark.asyncio
    async def test_introspect_4xx_returns_none(self):
        """A 4xx from the licenses endpoint = invalid token → None (→ 401)."""
        mock_resp = MagicMock()
        mock_resp.status_code = 403

        with patch("auth.middleware.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await _introspect_license("some-token")

        assert result is None

    @pytest.mark.asyncio
    async def test_introspect_5xx_raises_unavailable(self):
        """A 5xx is an availability failure → raise (→ 503), not a 401."""
        mock_resp = MagicMock()
        mock_resp.status_code = 503

        with patch("auth.middleware.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            with pytest.raises(LicenseServiceUnavailableError):
                await _introspect_license("some-token")

    @pytest.mark.asyncio
    async def test_introspect_network_error_raises_unavailable(self):
        """A transport error → raise (→ 503), not a misleading 401."""
        with patch("auth.middleware.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            with pytest.raises(LicenseServiceUnavailableError):
                await _introspect_license("error-token")


class TestAuthMiddlewareIntegration:
    """Integration tests for auth middleware with TestClient."""

    def test_public_endpoints_accessible(self):
        """Test that public endpoints don't require auth."""
        from main import app

        client = TestClient(app)

        # These should all be accessible without auth
        response = client.get("/")
        assert response.status_code == 200

        response = client.get("/health")
        assert response.status_code == 200

        response = client.get("/health/live")
        assert response.status_code == 200

        response = client.get("/docs")
        assert response.status_code == 200

    def test_protected_endpoint_without_auth(self):
        """Test that protected endpoints return 401 without auth."""
        from main import app

        client = TestClient(app, raise_server_exceptions=False)
        response = client.post("/api/scrape/url", json={"url": "https://example.com"})
        assert response.status_code == 401

    @patch("services.scraper_manager.ScraperManager.scrape_url")
    @patch("auth.middleware._introspect_license")
    def test_protected_endpoint_with_valid_license(
        self, mock_introspect: AsyncMock, mock_scrape: AsyncMock
    ):
        """Test that valid license grants access."""
        from main import app
        from services.base_scraper import ScrapeResult, ScrapeStatus

        mock_introspect.return_value = {"active": True, "subject_id": "test-tenant"}
        mock_scrape.return_value = ScrapeResult(
            url="https://example.com", status=ScrapeStatus.COMPLETED, html_content="<html>ok</html>"
        )

        client = TestClient(app, raise_server_exceptions=False)
        response = client.post(
            "/api/scrape/url",
            json={"url": "https://example.com"},
            headers={"X-Tenant-Id": "Bearer test-license-token"},
        )
        # Should not be 401 - may be other errors but not auth
        assert response.status_code != 401

    @patch("auth.middleware._introspect_license")
    def test_protected_endpoint_with_invalid_license(self, mock_introspect: AsyncMock):
        """Test that invalid license returns 401."""
        from main import app

        mock_introspect.return_value = {"active": False}

        client = TestClient(app, raise_server_exceptions=False)
        response = client.post(
            "/api/scrape/url",
            json={"url": "https://example.com"},
            headers={"X-Tenant-Id": "Bearer invalid-token"},
        )
        assert response.status_code == 401

    @patch("auth.middleware._introspect_license")
    def test_auth_401_carries_security_headers_and_timing(self, mock_introspect: AsyncMock):
        """Regression (round-N+7 middleware reorder): an auth-layer 401 must still
        pick up the baseline security headers (SecurityHeaders) AND the access-log
        timing header (RequestLogging). Before the reorder, AuthMiddleware wrapped
        both, so its short-circuit JSONResponse skipped them entirely.
        """
        from main import app

        mock_introspect.return_value = {"active": False}

        client = TestClient(app, raise_server_exceptions=False)
        response = client.post(
            "/api/scrape/url",
            json={"url": "https://example.com"},
            headers={"X-Tenant-Id": "Bearer invalid-token"},
        )
        assert response.status_code == 401
        # SecurityHeaders now wraps Auth → baseline headers present on the reject.
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert response.headers["X-Frame-Options"] == "DENY"
        # RequestLogging now wraps Auth → timing header present on the reject.
        assert "X-Processing-Time" in response.headers
        # RequestId was already outermost → request id still set (unchanged).
        assert "X-Request-ID" in response.headers

    @patch("auth.middleware._introspect_license")
    def test_auth_503_carries_security_headers_and_timing(self, mock_introspect: AsyncMock):
        """The 503 license-unavailable short-circuit also picks up the wrapping
        SecurityHeaders + RequestLogging after the reorder (and keeps Retry-After)."""
        from main import app

        mock_introspect.side_effect = LicenseServiceUnavailableError("licenses down")

        client = TestClient(app, raise_server_exceptions=False)
        response = client.post(
            "/api/scrape/url",
            json={"url": "https://example.com"},
            headers={"X-Tenant-Id": "Bearer any-token"},
        )
        assert response.status_code == 503
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert "X-Processing-Time" in response.headers
        assert response.headers.get("Retry-After") == "5"

    @patch("auth.middleware._introspect_license")
    def test_license_missing_tenant_context(self, mock_introspect: AsyncMock):
        """Test that license with no subject_id/tenant_id returns 401."""
        from main import app

        # Active but missing tenant context
        mock_introspect.return_value = {"active": True}

        client = TestClient(app, raise_server_exceptions=False)
        response = client.post(
            "/api/scrape/url",
            json={"url": "https://example.com"},
            headers={"X-Tenant-Id": "Bearer token-no-tenant"},
        )
        assert response.status_code == 401
        body = response.json()
        assert body["error"]["message"] == "License token missing tenant context"

    @patch("auth.middleware._introspect_license")
    def test_license_introspect_returns_none(self, mock_introspect: AsyncMock):
        """Test that None introspect result (network error) returns 401."""
        from main import app

        mock_introspect.return_value = None

        client = TestClient(app, raise_server_exceptions=False)
        response = client.post(
            "/api/scrape/url",
            json={"url": "https://example.com"},
            headers={"X-Tenant-Id": "Bearer bad-token"},
        )
        assert response.status_code == 401

    @patch("auth.middleware._introspect_license")
    def test_license_token_stashed_on_request_state(self, mock_introspect: AsyncMock):
        """Valid license-token path stores the raw bearer on request.state.license_token.

        Downstream handlers (router) need this to forward the token to the
        secrets service when fetching URL credentials.
        """
        from fastapi import FastAPI, Request

        mock_introspect.return_value = {"active": True, "subject_id": "tenant-x"}

        probe = FastAPI()
        probe.add_middleware(AuthMiddleware, protected_prefixes=["/probe"])

        @probe.get("/probe")
        async def _probe(request: Request) -> dict[str, object]:
            return {
                "license_token": getattr(request.state, "license_token", "MISSING"),
                "tenant_id": getattr(request.state, "tenant_id", None),
            }

        client = TestClient(probe, raise_server_exceptions=False)
        response = client.get("/probe", headers={"X-Tenant-Id": "Bearer my-license-token"})

        assert response.status_code == 200
        body = response.json()
        assert body["license_token"] == "my-license-token"
        assert body["tenant_id"] == "tenant-x"

    @patch("auth.middleware._is_trusted_network", return_value=True)
    def test_license_token_is_none_for_trusted_network(self, _mock_trusted):
        """Trusted-network callers (no bearer) get ``license_token = None``.

        The router then forwards plain ``X-Tenant-Id: <tenant>`` to the
        secrets service, which accepts this on the same trusted-network path.
        """
        from fastapi import FastAPI, Request

        probe = FastAPI()
        probe.add_middleware(AuthMiddleware, protected_prefixes=["/probe"])

        @probe.get("/probe")
        async def _probe(request: Request) -> dict[str, object]:
            return {
                "license_token": getattr(request.state, "license_token", "MISSING"),
                "tenant_id": getattr(request.state, "tenant_id", None),
            }

        client = TestClient(probe, raise_server_exceptions=False)
        response = client.get("/probe", headers={"X-Tenant-Id": "my-tenant"})

        assert response.status_code == 200
        body = response.json()
        assert body["license_token"] is None
        assert body["tenant_id"] == "my-tenant"

    def test_middleware_dispatch_exception_returns_500(self):
        """Test that middleware exception handler returns 500."""
        from main import app

        # Patch _extract_license_token to raise an exception
        with patch.object(
            AuthMiddleware,
            "_extract_license_token",
            side_effect=RuntimeError("middleware crash"),
        ):
            client = TestClient(app, raise_server_exceptions=False)
            response = client.post(
                "/api/scrape/url",
                json={"url": "https://example.com"},
                headers={"X-Tenant-Id": "Bearer some-token"},
            )
            assert response.status_code == 500
            body = response.json()
            assert body["error"]["code"] == "AUTH_ERROR"

    @patch("services.scraper_manager.ScraperManager.scrape_url")
    @patch("auth.middleware._introspect_license")
    def test_valid_license_with_tenant_id_field(
        self, mock_introspect: AsyncMock, mock_scrape: AsyncMock
    ):
        """Test license with tenant_id (not subject_id) is accepted."""
        from main import app
        from services.base_scraper import ScrapeResult, ScrapeStatus

        mock_introspect.return_value = {"active": True, "tenant_id": "alt-tenant"}
        mock_scrape.return_value = ScrapeResult(
            url="https://example.com", status=ScrapeStatus.COMPLETED, html_content="<html>ok</html>"
        )

        client = TestClient(app, raise_server_exceptions=False)
        response = client.post(
            "/api/scrape/url",
            json={"url": "https://example.com"},
            headers={"X-Tenant-Id": "Bearer valid-tenant-id-token"},
        )
        assert response.status_code != 401


class TestBypassAuthDispatch:
    """Test the bypass_auth code path in AuthMiddleware.dispatch."""

    def test_bypass_auth_sets_test_user(self):
        """When bypass_auth=True, dispatch sets user to test_user and proceeds."""
        from fastapi import FastAPI, Request
        from fastapi.testclient import TestClient

        # Create a minimal app with bypass_auth=True
        test_app = FastAPI()
        test_app.add_middleware(
            AuthMiddleware,
            protected_prefixes=["/api"],
            bypass_auth=True,
        )

        @test_app.get("/api/test")
        async def test_endpoint(request: Request):
            return {
                "user_id": request.state.user["user_id"],
                "tenant_id": request.state.tenant_id,
            }

        client = TestClient(test_app)
        # No auth headers needed -- bypass_auth skips all auth
        response = client.get("/api/test")
        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == "test_user"
        assert data["tenant_id"] == "test_user"
