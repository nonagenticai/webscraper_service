"""Integration-shape tests for the license-token AuthMiddleware.

These tests exercise the middleware against a minimal isolated ASGI app —
complementary to the deeper unit tests in ``tests/test_auth.py`` (which mock
internals and exercise the live ``main.app`` with TestClient).

Coverage:
- public endpoints bypass auth
- ``/api/*`` from a trusted-network peer (``testclient``) bypasses with the
  default tenant when no Bearer token is sent
- ``/api/*`` with a valid Bearer token in ``X-Tenant-Id`` is admitted and the
  introspection result populates ``request.state``
- ``/api/*`` with an invalid Bearer token → 401 with the canonical envelope
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

import auth.middleware as auth_middleware
from auth.middleware import AuthMiddleware, _introspect_license, _license_cache


def _make_app() -> FastAPI:
    """Build a minimal app with one public + one protected route."""
    app = FastAPI()
    app.add_middleware(AuthMiddleware, protected_prefixes=["/api"])

    @app.get("/health")
    async def _health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/echo")
    async def _echo(request: Request) -> JSONResponse:
        return JSONResponse(
            {
                "tenant_id": getattr(request.state, "tenant_id", None),
                "license_token": getattr(request.state, "license_token", "MISSING"),
                "user": getattr(request.state, "user", None),
            }
        )

    return app


@pytest.fixture(autouse=True)
def _clear_license_cache() -> None:
    """Each test starts with a fresh introspection cache."""
    _license_cache.clear()
    auth_middleware._license_neg_cache.clear()
    auth_middleware._introspect_locks.clear()


@pytest.mark.asyncio
async def test_introspect_single_flight_cold_burst() -> None:
    """A cold-cache burst of concurrent calls for the same token collapses into
    exactly ONE upstream POST (the 2026-06-27 stampede regression)."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"active": True, "subject_id": "tenant-1"}

    with patch("auth.middleware.httpx.AsyncClient") as mock_cls:
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp
        mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

        results = await asyncio.gather(*[_introspect_license("tok") for _ in range(12)])

    assert all(r is not None and r["active"] for r in results)
    assert mock_client.post.call_count == 1  # 12-way burst → one introspection


@pytest.mark.asyncio
async def test_introspect_inactive_negative_cached() -> None:
    """active:false is held in the negative cache, so a second call within the
    TTL does not re-hit the licenses endpoint."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"active": False}

    with patch("auth.middleware.httpx.AsyncClient") as mock_cls:
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp
        mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

        first = await _introspect_license("tok")
        second = await _introspect_license("tok")

    assert first == {"active": False}
    assert second == first
    assert mock_client.post.call_count == 1


@pytest.fixture
async def client() -> AsyncGenerator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=_make_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


async def test_health_is_public(client: httpx.AsyncClient) -> None:
    resp = await client.get("/health")
    assert resp.status_code == 200


async def test_protected_route_from_trusted_network_bypasses_auth(
    client: httpx.AsyncClient,
) -> None:
    """ASGI TestClient peer (host == '127.0.0.1') is a trusted network."""
    resp = await client.get("/api/echo")
    assert resp.status_code == 200
    body = resp.json()
    # DEFAULT_TENANT_ID defaults to 'n8n-dev'; no Bearer token → license_token None.
    assert body["tenant_id"] == "n8n-dev"
    assert body["license_token"] is None


async def test_trusted_network_whitespace_tenant_falls_back_to_default(
    client: httpx.AsyncClient,
) -> None:
    """A whitespace-only X-Tenant-Id must NOT become an empty-string tenant."""
    resp = await client.get("/api/echo", headers={"X-Tenant-Id": "   "})
    assert resp.status_code == 200
    assert resp.json()["tenant_id"] == "n8n-dev"


async def test_protected_route_with_valid_license_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A valid Bearer token in X-Tenant-Id is introspected and admitted."""
    introspect_payload: dict[str, Any] = {
        "active": True,
        "tenant_id": "tenant-uuid-1",
        "subject_id": "subject-uuid-9",
        "license_id": "lic-77",
        "tenant_name": "acme",
    }

    async def _fake(token: str) -> dict[str, Any]:
        assert token == "abc.def.ghi"
        return introspect_payload

    monkeypatch.setattr(auth_middleware, "_introspect_license", _fake)

    transport = httpx.ASGITransport(app=_make_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get(
            "/api/echo",
            headers={"X-Tenant-Id": "Bearer abc.def.ghi"},
        )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    # tenant_id wins over subject_id: they are DIFFERENT fields upstream.
    assert body["tenant_id"] == "tenant-uuid-1"
    assert body["user"]["tenant_id"] == "tenant-uuid-1"
    # The subject is kept on the principal, never used as the tenant.
    assert body["user"]["sub"] == "subject-uuid-9"
    assert body["user"]["user_id"] == "subject-uuid-9"
    assert body["license_token"] == "abc.def.ghi"


async def _echo_with_licence(
    monkeypatch: pytest.MonkeyPatch, payloads: dict[str, dict[str, Any]], token: str
) -> dict[str, Any]:
    async def _fake(tok: str) -> dict[str, Any]:
        return payloads[tok]

    monkeypatch.setattr(auth_middleware, "_introspect_license", _fake)
    transport = httpx.ASGITransport(app=_make_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/api/echo", headers={"X-Tenant-Id": f"Bearer {token}"})
    assert resp.status_code == 200, resp.text
    return resp.json()


async def test_one_subject_two_tenants_resolve_to_different_tenants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: one subject holding licences in two tenants must be
    attributed to two DIFFERENT tenants. Deriving the tenant from subject_id
    collapsed both onto the subject — a cross-tenant attribution."""
    payloads = {
        "tok-a": {"active": True, "subject_id": "subj-shared", "tenant_id": "tenant-a"},
        "tok-b": {"active": True, "subject_id": "subj-shared", "tenant_id": "tenant-b"},
    }
    a = await _echo_with_licence(monkeypatch, payloads, "tok-a")
    b = await _echo_with_licence(monkeypatch, payloads, "tok-b")
    assert a["tenant_id"] == "tenant-a"
    assert b["tenant_id"] == "tenant-b"
    assert a["tenant_id"] != b["tenant_id"]
    assert a["user"]["sub"] == b["user"]["sub"] == "subj-shared"


async def test_subject_only_licence_falls_back_to_subject(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A licence with no tenant_id still authenticates, on its subject."""
    body = await _echo_with_licence(
        monkeypatch, {"tok": {"active": True, "subject_id": "subj-only"}}, "tok"
    )
    assert body["tenant_id"] == "subj-only"
    assert body["user"]["sub"] == "subj-only"


async def test_protected_route_with_invalid_license_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An inactive introspection result returns 401 with the envelope shape."""

    async def _fake(_token: str) -> dict[str, Any]:
        return {"active": False}

    monkeypatch.setattr(auth_middleware, "_introspect_license", _fake)

    transport = httpx.ASGITransport(app=_make_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get(
            "/api/echo",
            headers={"X-Tenant-Id": "Bearer dud-token"},
        )

    assert resp.status_code == 401
    body = resp.json()
    assert body["error"]["code"] == "AUTH_REQUIRED"
    assert body["error"]["status"] == 401
    assert resp.headers["WWW-Authenticate"] == "Bearer"


async def test_protected_route_missing_tenant_in_introspection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Active token but no tenant_id and no subject_id → 401."""

    async def _fake(_token: str) -> dict[str, Any]:
        return {"active": True}

    monkeypatch.setattr(auth_middleware, "_introspect_license", _fake)

    transport = httpx.ASGITransport(app=_make_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get(
            "/api/echo",
            headers={"X-Tenant-Id": "Bearer no-tenant"},
        )

    assert resp.status_code == 401
    assert "tenant" in resp.json()["error"]["message"].lower()


async def test_license_service_unavailable_returns_503(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An upstream availability failure surfaces as a retryable 503, NOT a
    misleading 401 (the spurious-401 class). Distinct code + Retry-After."""

    async def _fake(_token: str) -> dict[str, Any]:
        raise auth_middleware.LicenseServiceUnavailableError("licenses down")

    monkeypatch.setattr(auth_middleware, "_introspect_license", _fake)

    transport = httpx.ASGITransport(app=_make_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/api/echo", headers={"X-Tenant-Id": "Bearer any-token"})

    assert resp.status_code == 503
    body = resp.json()
    assert body["error"]["code"] == "LICENSE_UNAVAILABLE"
    assert body["error"]["status"] == 503
    assert resp.headers.get("Retry-After") == "5"


def test_positive_expiry_capped_by_token_exp() -> None:
    """The positive-cache lifetime is pulled in by the token's own exp."""
    import time

    now = time.time()
    # exp 30s out → expiry ~now+30, well under the 5-min hard cap.
    soon = auth_middleware._positive_expiry({"active": True, "exp": now + 30})
    assert now + 25 < soon < now + 35
    # exp far in the future → clamped to the 5-min hard cap.
    capped = auth_middleware._positive_expiry({"active": True, "exp": now + 99999})
    assert capped <= now + auth_middleware._LICENSE_CACHE_MAX_TTL + 1
    # no exp → hard cap.
    default = auth_middleware._positive_expiry({"active": True})
    assert now + 290 < default <= now + 301
    # expires_in honored too.
    via_in = auth_middleware._positive_expiry({"active": True, "expires_in": 20})
    assert now + 15 < via_in < now + 25


async def test_expired_positive_cache_entry_is_a_miss(monkeypatch: pytest.MonkeyPatch) -> None:
    """A positive entry past its exp-capped expiry must re-introspect, not serve stale."""
    import hashlib
    import time

    token = "exp-token"
    cache_key = f"lic:{hashlib.sha256(token.encode()).hexdigest()[:16]}"
    # Seed an already-expired positive entry.
    _license_cache[cache_key] = ({"active": True, "subject_id": "old"}, time.time() - 1)

    calls = {"n": 0}

    async def _uncached(_t: str, _k: str) -> dict[str, Any]:
        calls["n"] += 1
        return {"active": True, "subject_id": "fresh"}

    monkeypatch.setattr(auth_middleware, "_introspect_license_uncached", _uncached)
    result = await _introspect_license(token)
    assert result["subject_id"] == "fresh"
    assert calls["n"] == 1


def test_introspect_locks_is_bounded() -> None:
    """The single-flight lock store is a bounded cache, not an unbounded dict."""
    from cachetools import TTLCache

    assert isinstance(auth_middleware._introspect_locks, TTLCache)
    assert auth_middleware._introspect_locks.maxsize <= 4096
