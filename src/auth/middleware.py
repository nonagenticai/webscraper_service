"""License Token Authentication Middleware.

Uses license token (`X-Tenant-Id: Bearer <token>`) as the ONLY auth method.
The ``Authorization`` header is reserved for the caller's own auth (e.g. an
upstream service forwarding a Keycloak JWT) and is never read here.

Successful introspection responses are cached for 5 minutes by SHA-256 prefix
of the token (TTLCache).
"""

import asyncio
import hashlib
import ipaddress
import logging
import os
import time
from collections.abc import Callable
from typing import Any, cast

import httpx
from cachetools import TTLCache
from fastapi import Request, Response, status
from starlette.middleware.base import BaseHTTPMiddleware

from .tenant import (
    ALLOW_TRUSTED_NETWORK_DEFAULT_TENANT,
    DEFAULT_TENANT_ID,
    log_tenant_fallback,
)

logger = logging.getLogger(__name__)

# Positive introspection cache. Hard upper bound 5 min; the per-entry expiry
# (capped by the token's own ``exp``/``expires_in``) is enforced on read so a
# token expiring sooner than 5 min isn't accepted for the full window.
_license_cache: TTLCache = TTLCache(maxsize=1000, ttl=300)
_LICENSE_CACHE_MAX_TTL = 300.0
# Negative cache for ``active: false`` so a revoked token can't pin every
# request against the licenses endpoint. Short TTL so re-issuance is picked up.
_license_neg_cache: TTLCache = TTLCache(
    maxsize=1000, ttl=int(os.getenv("LICENSE_INTROSPECT_NEG_TTL", "60"))
)

# Per-token single-flight locks. A cold-cache burst of concurrent requests for
# the SAME token would otherwise each miss the cache and independently POST the
# licenses endpoint; on a CPU-saturated box that stampede starves the loop and
# trips the introspect timeout → spurious 401 (the 2026-06-27 TTS bug class).
# The lock collapses the burst into one upstream call; the rest read the cache.
# Bounded (TTL/LRU) so it can't grow one Lock per distinct token forever — an
# evicted lock at worst costs one extra upstream call (correctness preserved).
_introspect_locks: TTLCache = TTLCache(maxsize=2048, ttl=600)


class LicenseServiceUnavailableError(Exception):
    """The licenses endpoint was unreachable / timed out / returned 5xx.

    Distinct from an *inactive* token: an availability failure must surface as a
    retryable 503, not a misleading 401 ("invalid token")."""


# License introspection endpoint — no hardcoded fallback. Production
# deployments MUST set LICENSE_INTROSPECT_URL via env (gitops ConfigMap);
# missing env in production is enforced by Settings.model_post_init.
LICENSE_INTROSPECT_URL = os.getenv("LICENSE_INTROSPECT_URL", "")

# Wall-clock budget for a single introspection POST. Default 10s (was a
# hardcoded 2s): on a CPU-saturated host the loop can starve long enough that a
# tighter budget fires before the licenses endpoint replies → spurious 401.
LICENSE_INTROSPECT_TIMEOUT = float(os.getenv("LICENSE_INTROSPECT_TIMEOUT", "10.0"))

PUBLIC_PATHS: list[str] = [
    "/",
    "/health",
    "/health/ready",
    "/health/live",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/favicon.ico",
]

# Re-exported (not re-read) so `auth.middleware.DEFAULT_TENANT_ID` keeps working
# for existing callers and tests. `auth/tenant.py` is the only module that reads
# the env var — see its docstring for why three independent readers was a
# mis-billing hazard rather than a style nit.
_TRUSTED_CIDRS_RAW = os.getenv(
    "TRUSTED_CIDRS",
    "127.0.0.1/32,::1/128,192.168.0.0/16,172.16.0.0/12,10.0.0.0/8",
)
TRUSTED_NETWORKS: list[ipaddress.IPv4Network | ipaddress.IPv6Network] = [
    ipaddress.ip_network(cidr.strip(), strict=False)
    for cidr in _TRUSTED_CIDRS_RAW.split(",")
    if cidr.strip()
]


def _is_trusted_network(host: str | None) -> bool:
    """Check if the client IP is within a trusted CIDR range."""
    if not host:
        return False
    try:
        addr = ipaddress.ip_address(host)
        return any(addr in net for net in TRUSTED_NETWORKS)
    except ValueError:
        return False


def _positive_expiry(data: dict[str, Any]) -> float:
    """Wall-clock instant after which a cached *active* license must be re-checked.

    Capped at ``_LICENSE_CACHE_MAX_TTL`` (5 min) but pulled earlier by the
    token's own ``exp`` (unix seconds) or ``expires_in`` (seconds) so a token
    expiring sooner is not honored for the full window.
    """
    now = time.time()
    hard = now + _LICENSE_CACHE_MAX_TTL
    exp = data.get("exp")
    if isinstance(exp, int | float) and not isinstance(exp, bool) and exp > 0:
        return min(hard, float(exp))
    expires_in = data.get("expires_in")
    if isinstance(expires_in, int | float) and not isinstance(expires_in, bool) and expires_in > 0:
        return min(hard, now + float(expires_in))
    return hard


def _cached_license(cache_key: str) -> dict[str, Any] | None:
    """Return a positive- or negative-cache hit for ``cache_key`` (or ``None``).

    Positive entries carry an explicit ``expires_at`` (capped by the token's
    own ``exp``); a past-expiry entry is treated as a miss so the caller
    re-introspects under the single-flight lock.
    """
    entry = _license_cache.get(cache_key)
    if entry is not None:
        data, expires_at = entry
        if time.time() < expires_at:
            return cast(dict[str, Any], data)
    neg = _license_neg_cache.get(cache_key)
    if neg is not None:
        return cast(dict[str, Any], neg)
    return None


async def _introspect_license(token: str) -> dict[str, Any] | None:
    """Introspect license token to get tenant context.

    ``active`` payloads are cached 5 min; ``active: false`` payloads are cached
    briefly in the negative cache. A cold-cache burst for the same token is
    collapsed into a single upstream POST via a per-token single-flight lock.
    """
    cache_key = f"lic:{hashlib.sha256(token.encode()).hexdigest()[:16]}"
    hit = _cached_license(cache_key)
    if hit is not None:
        return hit

    # Cache miss: single-flight so a concurrent burst for the same cold token
    # collapses into ONE upstream introspection. Re-check inside the lock — a
    # prior waiter may have populated the cache while we were blocked.
    lock = _introspect_locks.setdefault(cache_key, asyncio.Lock())
    async with lock:
        hit = _cached_license(cache_key)
        if hit is not None:
            return hit
        return await _introspect_license_uncached(token, cache_key)


async def _introspect_license_uncached(token: str, cache_key: str) -> dict[str, Any] | None:
    """Perform the introspection POST and populate the caches.

    Always called under the per-token single-flight lock held by
    :func:`_introspect_license`.

    Returns the introspection payload on a 200 (cached: active → positive with
    exp-capped TTL, inactive → negative). Returns ``None`` for a 4xx (treated as
    an invalid token → 401). Raises :class:`LicenseServiceUnavailableError` for an
    availability failure (timeout / transport error / 5xx / unparseable body) so
    the caller can answer 503 rather than a misleading 401.
    """
    try:
        async with httpx.AsyncClient(timeout=LICENSE_INTROSPECT_TIMEOUT, verify=True) as client:
            resp = await client.post(LICENSE_INTROSPECT_URL, json={"token": token})
    except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPError) as e:
        logger.warning(f"License introspection upstream error: {e}")
        raise LicenseServiceUnavailableError(str(e)) from e

    if resp.status_code == 200:
        try:
            data: dict[str, Any] = resp.json()
        except ValueError as e:
            logger.warning(f"License introspection returned an unparseable body: {e}")
            raise LicenseServiceUnavailableError("unparseable introspection body") from e
        if data.get("active"):
            _license_cache[cache_key] = (data, _positive_expiry(data))
        else:
            _license_neg_cache[cache_key] = data
        return data

    if resp.status_code >= 500:
        logger.warning(f"License endpoint returned {resp.status_code}")
        raise LicenseServiceUnavailableError(f"licenses endpoint returned {resp.status_code}")

    # 4xx → the token itself is rejected by the licenses endpoint → 401 (no cache).
    logger.warning(f"License introspection non-200: {resp.status_code}")
    return None


class AuthMiddleware(BaseHTTPMiddleware):
    """License Token Authentication Middleware."""

    def __init__(
        self,
        app: Any,
        protected_prefixes: list[str] | None = None,
        bypass_auth: bool = False,
    ):
        super().__init__(app)
        self.protected_prefixes = protected_prefixes or []
        self.bypass_auth = bypass_auth

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Authenticate, then hand the request on.

        ``call_next`` is deliberately OUTSIDE the try: with it inside, every
        exception raised by a route — a genuine bug in the scrape path — was
        caught here and answered as "Authentication service error", with no
        traceback and a misleading cause. Only the authentication work itself
        is guarded.
        """
        try:
            decision = await self._authorize(request)
        except Exception as e:
            logger.error(f"Authentication middleware error: {e}", exc_info=True)
            return self._internal_error_response("Authentication service error")

        if decision is not None:
            return decision
        return cast(Response, await call_next(request))

    async def _authorize(self, request: Request) -> Response | None:
        """Resolve the caller's identity onto ``request.state``.

        Returns:
            A Response to short-circuit with (rejection/unavailable), or None
            when the request is authenticated and may proceed.
        """
        path = request.url.path.rstrip("/") or "/"

        # Skip auth for public/non-protected endpoints
        if self._is_public_path(path) or not self._is_protected_path(path):
            return None

        # Skip auth if bypassed (for testing)
        if self.bypass_auth:
            request.state.user = {"user_id": "test_user", "tenant_id": "test_user"}
            request.state.tenant_id = "test_user"
            request.state.license_token = None
            return None

        # Extract token from X-Tenant-Id header (Bearer format)
        token = self._extract_license_token(request)
        x_tenant_id = request.headers.get("X-Tenant-Id")

        # If a token is provided, always validate it (even from trusted networks)
        if token:
            try:
                license_result = await _introspect_license(token)
            except LicenseServiceUnavailableError:
                # Upstream down/timeout/5xx — retryable, NOT an invalid token.
                return self._service_unavailable_response("License service unavailable")
            if not license_result or not license_result.get("active"):
                return self._unauthorized_response("Invalid or expired license token")

            # tenant_id FIRST. The licences service returns subject_id and
            # tenant_id as DIFFERENT fields (a subject is the licence holder, and
            # one subject can hold licences in several tenants), so preferring
            # subject_id attributed every request to the wrong tenant. subject_id
            # is only a fallback for a licence that carries no tenant_id at all.
            # Reference: requirements_management 43dc8f1 src/auth/middleware.py.
            tenant_id = license_result.get("tenant_id") or license_result.get("subject_id")
            if not tenant_id:
                return self._unauthorized_response("License token missing tenant context")
            # The principal keeps the subject separately; it never stands in for
            # the tenant when a tenant_id is present.
            subject_id = license_result.get("subject_id") or tenant_id

            request.state.user = {
                "user_id": str(subject_id),
                "tenant_id": str(tenant_id),
                "sub": str(subject_id),
            }
            request.state.tenant_id = str(tenant_id)
            request.state.license_token = token
            logger.debug(f"Authenticated via license token: {tenant_id}")
            return None

        # No token — check if request is from a trusted network
        client_host = request.client.host if request.client else None
        if _is_trusted_network(client_host):
            asserted = (x_tenant_id or "").strip()
            if asserted:
                # The peer named its own tenant. Already attributed, so it is
                # not what the burn-down is about and the gate does not touch it.
                tenant_id = asserted
            else:
                # Nobody named a tenant: this request is about to be billed to
                # whoever DEFAULT_TENANT_ID happens to be — a real customer, not
                # a placeholder. Gated so it can eventually be refused, and
                # logged EVERY time so the callers that must be credentialed
                # first can actually be enumerated. See auth/tenant.py.
                if not ALLOW_TRUSTED_NETWORK_DEFAULT_TENANT:
                    return self._unauthorized_response("Missing tenant context")
                tenant_id = DEFAULT_TENANT_ID
                log_tenant_fallback(client_host, request.url.path)
            request.state.user = {
                "user_id": str(tenant_id),
                "tenant_id": str(tenant_id),
                "sub": str(tenant_id),
            }
            request.state.tenant_id = str(tenant_id)
            request.state.license = None
            request.state.license_token = None
            logger.debug(f"Trusted network access from {client_host}: tenant={tenant_id}")
            return None

        # Not trusted, no token — reject
        return self._unauthorized_response("Missing authentication token")

    def _is_public_path(self, path: str) -> bool:
        """Check if endpoint is public."""
        if path in PUBLIC_PATHS:
            return True
        public_prefixes = ["/docs", "/redoc", "/static", "/health"]
        return any(path.startswith(prefix) for prefix in public_prefixes)

    def _is_protected_path(self, path: str) -> bool:
        """Check if path requires authentication."""
        return any(path == prefix or path.startswith(prefix) for prefix in self.protected_prefixes)

    def _extract_license_token(self, request: Request) -> str | None:
        """Extract Bearer token from X-Tenant-Id header."""
        x_tenant_id = request.headers.get("X-Tenant-Id")
        if x_tenant_id and x_tenant_id.lower().startswith("bearer "):
            return x_tenant_id[7:].strip()
        return None

    def _unauthorized_response(self, detail: str) -> Response:
        """Return 401 Unauthorized response."""
        from fastapi.responses import JSONResponse

        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"error": {"code": "AUTH_REQUIRED", "message": detail, "status": 401}},
            headers={"WWW-Authenticate": "Bearer"},
        )

    def _internal_error_response(self, detail: str) -> Response:
        """Return 500 Internal Server Error response."""
        from fastapi.responses import JSONResponse

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": {"code": "AUTH_ERROR", "message": detail, "status": 500}},
        )

    def _service_unavailable_response(self, detail: str) -> Response:
        """Return 503 when the license service is unreachable (retryable)."""
        from fastapi.responses import JSONResponse

        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"error": {"code": "LICENSE_UNAVAILABLE", "message": detail, "status": 503}},
            headers={"Retry-After": "5"},
        )
