"""Configuration settings for webscraper service.

This module provides centralized configuration management using Pydantic
settings, with support for environment variables and .env files.
"""

from functools import lru_cache
from typing import Annotated, ClassVar

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from auth import tenant as _tenant


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    All settings can be overridden via environment variables or a .env file.
    Environment variable names are case-insensitive.

    Attributes:
        service_name: Name of the service for identification.
        version: Semantic version string.
        environment: Deployment environment (development, staging, production).
        host: Host address to bind the server.
        port: Port number for the server (1-65535).
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        playwright_headless: Whether to run Playwright browser in headless mode.
        scraping_default_delay: Default delay between requests in seconds.
        scraping_user_agent: Default user agent string for requests.
        scraping_max_concurrent: Maximum concurrent scraping operations.
    """

    service_name: str = "webscraper-service"
    version: str = "1.0.0"
    environment: str = "development"

    # Default sourced from auth.tenant rather than restating the literal here.
    # pydantic-settings still maps the DEFAULT_TENANT_ID env var onto this field
    # (case-insensitive), so an explicit env value wins exactly as before — this
    # only removes the third independent copy of the default, which could drift
    # away from the one the auth path bills against. See auth/tenant.py.
    default_tenant_id: str = Field(default_factory=lambda: _tenant.DEFAULT_TENANT_ID)

    host: str = "0.0.0.0"  # nosec B104 - intentional for Docker containers
    port: Annotated[int, Field(ge=1, le=65535)] = 9119
    log_level: str = "INFO"

    playwright_headless: bool = True

    scraping_default_delay: float = 1.0
    scraping_user_agent: str = "WebScraper-Service/1.0 (+https://example.com/bot)"
    scraping_max_concurrent: int = 16

    # Hard cap on a single fetched HTML body. The httpx scraper streams the
    # response and aborts past this (mirrors the per-image MAX_IMAGE_BYTES guard)
    # so a server returning a multi-GB body can't OOM the (single-replica) pod —
    # a buffered body would be resident AND multiplied by the BeautifulSoup +
    # trafilatura parse trees long before the response-side output clamp ever
    # runs, and the per-chunk read timeout never bounds total transfer. Scrapy
    # enforces the same ceiling via DOWNLOAD_MAXSIZE. Env-tunable.
    max_page_bytes: int = 10 * 1024 * 1024  # 10 MB

    # Scrapy crawl bounds (enforced inside the crawl subprocess). DOWNLOAD_TIMEOUT
    # only bounds a single request; with follow_links=True a crawl is otherwise
    # unbounded in wall-clock AND page count (every followed page is accumulated
    # in the child's memory before all but the first are discarded). These cap the
    # whole crawl via CLOSESPIDER_TIMEOUT / CLOSESPIDER_PAGECOUNT; the async await
    # is additionally wrapped in asyncio.wait_for as a hung-subprocess backstop.
    scrapy_crawl_timeout: float = 120.0  # CLOSESPIDER_TIMEOUT, seconds
    scrapy_max_pages: int = 50  # CLOSESPIDER_PAGECOUNT, max responses per crawl

    # SSRF guard: by default the scraper refuses to fetch loopback, link-local
    # (incl. cloud-metadata 169.254.x), private/RFC1918, and *.internal targets,
    # so an authenticated caller can't pivot to internal infrastructure.
    # ssrf_allowed_hosts is an exact-host escape hatch (e.g. a trusted internal
    # host that legitimately needs scraping, or a loopback test server).
    ssrf_protection_enabled: bool = True
    ssrf_allowed_hosts: list[str] = Field(default_factory=list)

    # Optional outbound HTTP(S) proxy for ALL scrape egress (httpx, Playwright,
    # Scrapy, curl_cffi). A single ops-controlled egress proxy — NOT a per-request
    # field (a caller-supplied proxy would be an SSRF-bypass + abuse surface). The
    # SSRF guard still validates the *target* host regardless of the proxy. Empty
    # (default) = direct connection. Example: "http://proxy.internal:3128".
    outbound_proxy: str | None = None

    # Result cache: short-TTL in-memory cache of COMPLETED single-URL scrapes,
    # keyed by the request shape, to cut duplicate fetches in re-ingest/bulk
    # patterns. ``cache_ttl_seconds=0`` disables it; per-request ``no_cache=true``
    # bypasses it. Entries hold the full ScrapeResponse (incl. html), so keep
    # ``cache_max_size`` modest on a memory-limited pod. Authed (auth_config)
    # scrapes are NEVER cached (no credential-bound cross-caller reuse).
    cache_ttl_seconds: int = 300
    cache_max_size: int = 128

    # Upstream service URLs — no hardcoded fallbacks. Production deployments
    # MUST supply these via env (gitops ConfigMap); the validator below
    # raises a loud RuntimeError on startup if any are missing in production.
    secrets_service_url: str = ""
    secrets_service_timeout: float = 10.0
    license_introspect_url: str = ""

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="ignore")

    _REQUIRED_IN_PRODUCTION: ClassVar[tuple[str, ...]] = (
        "secrets_service_url",
        "license_introspect_url",
    )

    def model_post_init(self, __context: object) -> None:
        if self.environment != "production":
            return
        missing = [n for n in self._REQUIRED_IN_PRODUCTION if not getattr(self, n)]
        if missing:
            raise RuntimeError(
                f"webscraper_service settings missing required fields in production: {missing}. "
                "Set them via env vars on the Deployment (see gitops-production)."
            )


@lru_cache
def get_settings() -> Settings:
    """Get cached settings singleton instance.

    Uses LRU cache to ensure only one Settings instance is created
    and reused throughout the application lifecycle.

    Returns:
        The application Settings instance.
    """
    return Settings()


def get_config() -> Settings:
    """Get settings instance (alias for get_settings).

    Returns:
        The application Settings instance.
    """
    return get_settings()
