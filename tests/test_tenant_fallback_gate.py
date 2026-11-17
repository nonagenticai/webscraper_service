"""Tests for the trusted-network default-tenant gate and its burn-down signal.

The fallback at ``auth/middleware.py`` serves a header-less request from a peer
inside TRUSTED_CIDRS (default: the whole pod CIDR) as DEFAULT_TENANT_ID. That
default is a real billable tenant, so every such request is billed to a customer
who did not make it. These tests pin the two things that make that fixable:

- ``ALLOW_TRUSTED_NETWORK_DEFAULT_TENANT`` exists, defaults to ON (flipping it
  off today would 401 live in-cluster callers), and actually refuses when off.
- A WARNING naming the peer and path is emitted on EVERY use, so the callers
  that must be credentialed before the flip can be enumerated from Loki.

Same isolated-ASGI-app shape as ``tests/test_auth_middleware.py``.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

import auth.middleware as auth_middleware
import auth.tenant as auth_tenant
from auth.middleware import AuthMiddleware


def _make_app() -> FastAPI:
    """Minimal app with one protected route that echoes the resolved tenant."""
    app = FastAPI()
    app.add_middleware(AuthMiddleware, protected_prefixes=["/api"])

    @app.get("/api/echo")
    async def _echo(request: Request) -> JSONResponse:
        return JSONResponse({"tenant_id": getattr(request.state, "tenant_id", None)})

    return app


@pytest.fixture
async def client() -> AsyncGenerator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=_make_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# ── the gate ────────────────────────────────────────────────────────────────


def test_gate_defaults_to_on() -> None:
    """MUST default to ON. Off-by-default would 401 every in-cluster caller that
    presents no credential today; closing it is a separate, measured gitops step
    gated on the TENANT_FALLBACK log going quiet."""
    assert auth_tenant.ALLOW_TRUSTED_NETWORK_DEFAULT_TENANT is True
    assert auth_middleware.ALLOW_TRUSTED_NETWORK_DEFAULT_TENANT is True


async def test_gate_open_serves_default_tenant(client: httpx.AsyncClient) -> None:
    """With the gate open (the default), current behaviour is unchanged."""
    resp = await client.get("/api/echo")
    assert resp.status_code == 200
    assert resp.json()["tenant_id"] == auth_tenant.DEFAULT_TENANT_ID


async def test_gate_closed_rejects_uncredentialed_trusted_peer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the gate closed, a trusted peer naming no tenant is 401'd rather than
    billed to DEFAULT_TENANT_ID. This is the end state the warning burns down to."""
    monkeypatch.setattr(auth_middleware, "ALLOW_TRUSTED_NETWORK_DEFAULT_TENANT", False)

    transport = httpx.ASGITransport(app=_make_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/api/echo")

    assert resp.status_code == 401
    body = resp.json()
    assert body["error"]["code"] == "AUTH_REQUIRED"
    assert "tenant" in body["error"]["message"].lower()


async def test_gate_closed_still_honors_an_asserted_tenant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The gate covers ONLY the default-tenant arm. A trusted peer that names its
    own tenant is already attributed, so closing the gate must not 401 it —
    otherwise the flip breaks correctly-behaving callers too."""
    monkeypatch.setattr(auth_middleware, "ALLOW_TRUSTED_NETWORK_DEFAULT_TENANT", False)

    transport = httpx.ASGITransport(app=_make_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/api/echo", headers={"X-Tenant-Id": "tenant-abc"})

    assert resp.status_code == 200
    assert resp.json()["tenant_id"] == "tenant-abc"


# ── the burn-down signal (the point of the change) ──────────────────────────


async def test_fallback_emits_warning_naming_peer_and_path(
    client: httpx.AsyncClient, caplog: pytest.LogCaptureFixture
) -> None:
    """Every use of the fallback logs at WARNING with the peer and path.

    WARNING, not debug: the pre-existing debug line on this branch never reached
    Loki (the live Deployment sets LOG_LEVEL=INFO), which is why nobody could
    enumerate these callers. Without this line the gate is unflippable.
    """
    with caplog.at_level(logging.WARNING, logger="auth.tenant"):
        resp = await client.get("/api/echo")

    assert resp.status_code == 200
    assert "TENANT_FALLBACK" in caplog.text
    assert "peer=127.0.0.1" in caplog.text
    assert "path=/api/echo" in caplog.text
    assert any(r.levelno == logging.WARNING for r in caplog.records)


async def test_no_fallback_warning_when_tenant_is_asserted(
    client: httpx.AsyncClient, caplog: pytest.LogCaptureFixture
) -> None:
    """An attributed request must NOT be logged. A signal that fires on traffic
    that is already fine cannot be burned down to zero."""
    with caplog.at_level(logging.WARNING, logger="auth.tenant"):
        resp = await client.get("/api/echo", headers={"X-Tenant-Id": "tenant-abc"})

    assert resp.status_code == 200
    assert "TENANT_FALLBACK" not in caplog.text


async def test_fallback_increments_prometheus_counter(client: httpx.AsyncClient) -> None:
    """The same event is counted so an alert can watch it reach zero."""
    before = auth_tenant.TENANT_FALLBACK_TOTAL._value.get()
    resp = await client.get("/api/echo")
    assert resp.status_code == 200
    assert auth_tenant.TENANT_FALLBACK_TOTAL._value.get() == before + 1


def test_fallback_counter_is_on_the_default_registry() -> None:
    """routers/metrics.py serves generate_latest() over the DEFAULT registry, so
    the counter must be registered there or /metrics never shows it."""
    from prometheus_client import generate_latest

    assert "webscraper_tenant_fallback_total" in generate_latest().decode()


# ── single-reader consolidation ─────────────────────────────────────────────


def test_default_tenant_id_has_exactly_one_env_reader() -> None:
    """Only auth/tenant.py may read DEFAULT_TENANT_ID from the environment.

    It used to be read independently in auth/middleware.py, auth/dependencies.py
    and config.py. Three readers of one billing-relevant env var is three chances
    to drift, and a drifted default does not crash — it bills the wrong tenant.

    Matched over the AST, not the file text: the comments that record the removal
    quote the very call they removed, so a substring grep re-flags every module it
    just cleaned up (it did, on the first run of this test).
    """
    import ast
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "src"
    readers = []
    for path in sorted(src.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if name not in {"getenv", "environ"}:
                continue
            first = node.args[0]
            if isinstance(first, ast.Constant) and first.value == "DEFAULT_TENANT_ID":
                readers.append(path.relative_to(src).as_posix())
                break

    assert sorted(set(readers)) == ["auth/tenant.py"], readers


def test_all_consumers_share_one_value() -> None:
    """The three former readers now resolve to the same object."""
    import auth.dependencies as auth_deps
    from config import Settings

    assert auth_middleware.DEFAULT_TENANT_ID is auth_tenant.DEFAULT_TENANT_ID
    assert auth_deps.DEFAULT_TENANT_ID is auth_tenant.DEFAULT_TENANT_ID
    assert Settings().default_tenant_id == auth_tenant.DEFAULT_TENANT_ID


def test_default_is_not_the_nil_uuid() -> None:
    """billing_service quarantines nil-tenant events into a dead-letter stream
    that skips both the token rollup and the budget debit, converting mis-billing
    into un-billing. Never 'fix' attribution by defaulting to the nil UUID."""
    assert auth_tenant.DEFAULT_TENANT_ID != "00000000-0000-0000-0000-000000000000"


def test_default_value_is_unchanged_by_this_pr() -> None:
    """The value is deliberately NOT changed here. Adding the gate and changing
    the default at once would make any billing regression unattributable to one
    of the two. 'n8n-dev' is a placeholder in appearance only — billing's
    canonical_tenant_id resolves it to a real tenant and bills it; the live
    Deployment overrides it with a real tenant UUID regardless."""
    assert auth_tenant.DEFAULT_TENANT_ID == "n8n-dev"
