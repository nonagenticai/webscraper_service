"""WebScraper Service - Main FastAPI Application.

Standalone web scraping microservice with Playwright and Scrapy support.
Provides REST API for single URL and bulk scraping operations.

Auth: license token via ``X-Tenant-Id: Bearer <token>``, validated against the
license introspection endpoint. Trusted-network callers (RFC 1918 / loopback /
ASGI TestClient peer) bypass token requirement with the configured default
tenant.
"""

import logging
import os
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from auth.middleware import PUBLIC_PATHS, AuthMiddleware
from config import get_settings
from middleware.logging import RequestLoggingMiddleware
from middleware.request_id import RequestIdMiddleware
from middleware.security_headers import SecurityHeadersMiddleware
from routers import health, metrics, scrape
from services.scraper_manager import get_scraper_manager

settings = get_settings()

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_CORS_ORIGINS: list[str] = [
    "http://localhost:8921",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

PROTECTED_PREFIXES: list[str] = ["/api"]

ENABLE_AUTH = os.getenv("ENABLE_AUTH", "true").lower() == "true"
LEAK_EXCEPTION_DETAILS = os.getenv(
    "LEAK_EXCEPTION_DETAILS", ""
).lower() == "true" or settings.environment.lower() in {"development", "dev"}


def get_cors_config() -> tuple[list[str], bool]:
    """Get CORS configuration from environment.

    Parses CORS_ALLOW_ORIGINS to determine allowed origins. If '*' is present,
    credentials must be False per browser rules.
    """
    raw_origins = os.getenv("CORS_ALLOW_ORIGINS")
    if raw_origins:
        origins = [o.strip() for o in raw_origins.split(",") if o.strip()]
    else:
        origins = DEFAULT_CORS_ORIGINS

    allow_credentials = "*" not in origins
    if not allow_credentials:
        origins = ["*"]

    return origins, allow_credentials


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Manage application lifecycle for startup and shutdown events."""
    logger.info(f"Starting {settings.service_name} v{settings.version}")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Port: {settings.port}")

    scraper_manager = get_scraper_manager()
    logger.info("Scraper manager initialized")

    yield

    logger.info("Shutting down webscraper service...")
    await scraper_manager.close()
    logger.info("Scraper manager closed")


app = FastAPI(
    title="WebScraper Service",
    description=(
        "Web scraping microservice with multiple backends: Playwright (JS-rendered pages), "
        "Scrapy (bulk crawling), and httpx+BeautifulSoup (fast static pages). "
        "Auto mode selects the best backend per URL. "
        "Auth: license token via `X-Tenant-Id: Bearer <token>`, validated against "
        "LICENSE_INTROSPECT_URL. Trusted-network clients (RFC 1918) bypass token requirement."
    ),
    version=settings.version,
    lifespan=lifespan,
    redirect_slashes=False,
    docs_url="/docs",
    redoc_url="/redoc",
    swagger_ui_parameters={"persistAuthorization": True},
    # Fallback unique-ID generator. Every route still sets `operation_id=`
    # explicitly per OPERATION_ID_PLAN.md; this exists so a forgotten ID
    # degrades to `<tag>_<func>` (readable) instead of FastAPI's default
    # `<func>_<path>_<method>` (garbage in generated clients).
    generate_unique_id_function=lambda route: (
        f"{route.tags[0] if route.tags else 'default'}_{route.name}"
    ),
    openapi_tags=[
        {"name": "Health", "description": "Liveness, readiness, and root probes."},
        {
            "name": "Scrape",
            "description": (
                "Web scraping operations: single URL fetch, bulk crawl, multi-backend "
                "(Playwright / Scrapy / httpx) selection."
            ),
        },
    ],
)

# --- Middleware ---
# Starlette executes middleware in reverse-registration order: the last
# add_middleware() call is the outermost wrapper and runs first on the
# request path. Target request-side execution order:
#
#   RequestId → SecurityHeaders → RequestLogging → Auth → CORS → handler
#
# Rationale:
#   - RequestId outermost so every downstream layer (incl. auth 401/503s) can
#     reference request.state.request_id and the X-Request-ID response header
#     is always set.
#   - SecurityHeaders + RequestLogging wrap Auth (not just the handler) so that
#     auth-layer short-circuits — 401 (invalid/missing token) and 503 (license
#     service down), which `return` a JSONResponse without descending the rest
#     of the stack — STILL pick up the baseline security headers, an access-log
#     line, and the X-Processing-Time header. (They previously skipped both,
#     since Auth used to wrap SecurityHeaders/RequestLogging rather than nest
#     inside them.)
#   - Auth then short-circuits before CORS + the handler run.
cors_origins, allow_credentials = get_cors_config()
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-Processing-Time"],
)

if ENABLE_AUTH:
    app.add_middleware(AuthMiddleware, protected_prefixes=PROTECTED_PREFIXES)
    logger.info("Authentication middleware enabled")
else:
    logger.warning("Authentication middleware DISABLED (ENABLE_AUTH=false)")

app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(SecurityHeadersMiddleware)

app.add_middleware(RequestIdMiddleware)


# --- Global exception handlers ---
def _error_envelope(
    code: str,
    message: str,
    http_status: int,
    *,
    request_id: str,
    details: str | None = None,
) -> dict[str, Any]:
    """Build the standardized error envelope.

    Shape matches ``routers/responses.ErrorResponse``: a single ``error`` key
    with ``code``, ``message``, ``status`` (HTTP) plus the request-id and an
    ISO-8601 timestamp. ``details`` is only included when populated.
    """
    payload: dict[str, Any] = {
        "code": code,
        "message": message,
        "status": http_status,
        "request_id": request_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if details is not None:
        payload["details"] = details
    return {"error": payload}


def _http_exception_to_response(exc: StarletteHTTPException, request_id: str) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content=_error_envelope(
            f"HTTP_{exc.status_code}", str(exc.detail), exc.status_code, request_id=request_id
        ),
        headers=exc.headers,
    )


def _find_http_exception(exc: BaseException) -> HTTPException | None:
    """Walk nested ExceptionGroups to find the first HTTPException."""
    if isinstance(exc, HTTPException):
        return exc
    if isinstance(exc, BaseExceptionGroup):
        for sub in exc.exceptions:
            found = _find_http_exception(sub)
            if found is not None:
                return found
    return None


def _request_id_from(request: Request) -> str:
    return request.state.request_id if hasattr(request.state, "request_id") else str(uuid.uuid7())


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """Render HTTPException (incl. FastAPI's 404 for unknown routes) into the envelope."""
    return _http_exception_to_response(exc, _request_id_from(request))


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Render pydantic request-validation errors into the envelope (HTTP 422)."""
    request_id = _request_id_from(request)
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content=_error_envelope(
            "VALIDATION_ERROR",
            "Request validation failed",
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            request_id=request_id,
            details=str(exc.errors()) if LEAK_EXCEPTION_DETAILS else None,
        ),
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all for uncaught exceptions; returns the standardized envelope."""
    request_id: str = _request_id_from(request)

    http_exc = _find_http_exception(exc)
    if http_exc is not None:
        return _http_exception_to_response(http_exc, request_id)

    logger.error(f"unhandled_error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=_error_envelope(
            "INTERNAL_SERVER_ERROR",
            "An unexpected error occurred",
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            request_id=request_id,
            details=str(exc) if LEAK_EXCEPTION_DETAILS else None,
        ),
    )


# --- OpenAPI schema with per-path security ---
def custom_openapi() -> dict[str, Any]:
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
        tags=app.openapi_tags,
    )

    components = openapi_schema.setdefault("components", {})
    security_schemes = components.setdefault("securitySchemes", {})
    security_schemes.setdefault(
        "BearerAuth",
        {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "License",
            "description": (
                "License token. Transport: `X-Tenant-Id: Bearer <token>` "
                "(not the Authorization header — that is reserved for caller-side auth)."
            ),
        },
    )

    parameters = components.setdefault("parameters", {})
    parameters.setdefault(
        "TenantIdHeader",
        {
            "name": "X-Tenant-Id",
            "in": "header",
            "required": False,
            "description": "Tenant identifier (defaults to configured DEFAULT_TENANT_ID).",
            "schema": {"type": "string", "default": settings.default_tenant_id},
            "example": settings.default_tenant_id,
        },
    )

    for path, path_item in openapi_schema.get("paths", {}).items():
        normalized = path.rstrip("/") or "/"
        is_public = normalized in {"/", *PUBLIC_PATHS} or normalized.startswith(
            ("/docs", "/redoc", "/static", "/openapi.json", "/favicon.ico", "/health")
        )
        is_protected = any(
            normalized == prefix or normalized.startswith(prefix) for prefix in PROTECTED_PREFIXES
        )

        for operation in list(path_item.values()):
            if not isinstance(operation, dict):
                continue

            if is_protected and not is_public:
                operation.setdefault("security", [{"BearerAuth": []}])
            elif is_public:
                operation.pop("security", None)

            op_params = operation.setdefault("parameters", [])
            has_tenant = any(
                p.get("name", "").lower() == "x-tenant-id" and p.get("in") == "header"
                for p in op_params
            )
            if not has_tenant:
                op_params.append({"$ref": "#/components/parameters/TenantIdHeader"})

    app.openapi_schema = openapi_schema
    return app.openapi_schema


object.__setattr__(app, "openapi", custom_openapi)

app.include_router(health.router)
app.include_router(metrics.router)
app.include_router(scrape.router)


@app.get(
    "/",
    operation_id="service_root",
    tags=["Health"],
    summary="Service banner",
)
async def root() -> dict[str, object]:
    """Provide service information at the root endpoint (public)."""
    return {
        "service": settings.service_name,
        "version": settings.version,
        "status": "online",
        "endpoints": {
            "health": "/health",
            "scrape_url": "/api/scrape/url",
            "scrape_bulk": "/api/scrape/bulk",
            "docs": "/docs",
            "redoc": "/redoc",
        },
    }
