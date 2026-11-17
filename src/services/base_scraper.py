"""Base scraper class and common scraping utilities.

This module defines the abstract interface that all scrapers must implement,
along with shared data structures and utility methods for web scraping.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any
from urllib.parse import urljoin, urlparse

logger = logging.getLogger(__name__)


class ScraperType(StrEnum):
    """Available scraper backend types.

    Attributes:
        PLAYWRIGHT: Use Playwright for JavaScript rendering.
        SCRAPY: Use Scrapy for efficient bulk crawling.
        HTTPX: Use httpx + BeautifulSoup for fast static page scraping.
        CURL_CFFI: Use curl_cffi (Chrome TLS/JA3 impersonation) for static pages
            behind network-layer anti-bot blocks; no JavaScript.
        AUTO: Automatically select the best scraper.
    """

    PLAYWRIGHT = "playwright"
    SCRAPY = "scrapy"
    HTTPX = "httpx"
    CURL_CFFI = "curl_cffi"
    AUTO = "auto"


class ScrapeStatus(StrEnum):
    """Status of a scrape operation.

    Attributes:
        QUEUED: Operation is waiting to be processed.
        RUNNING: Operation is currently in progress.
        COMPLETED: Operation finished successfully.
        FAILED: Operation encountered an error.
        PAUSED: Operation is temporarily paused.
        CANCELLED: Operation was cancelled by user.
    """

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    CANCELLED = "cancelled"


@dataclass
class AuthConfig:
    """Service-layer authentication config.

    Holds *resolved* credentials (after the router fetched from the secrets
    service) plus the UI-flow hints from the request. The PlaywrightScraper
    uses this directly without any coupling to secrets/HTTP.

    Attributes:
        type: Auth flow selector — ``form``, ``storage_state``, ``bearer``, or ``basic``.
        username: Resolved username for form/basic auth.
        password: Resolved password for form/basic auth.
        token: Resolved bearer token.
        storage_state: Playwright storageState JSON for session injection.
        login_url: Optional explicit login URL (overrides redirect-detection).
        username_selector: CSS selector for the username input (None → defaults).
        password_selector: CSS selector for the password input (None → defaults).
        submit_selector: CSS selector for the submit button (None → defaults).
        wait_for_url: URL pattern to wait for after submit.
        wait_for_selector: CSS selector to wait for after submit.
        wait_timeout_ms: Max milliseconds to wait for post-login state.
    """

    type: str = "form"
    username: str = ""
    password: str = ""
    token: str = ""
    storage_state: dict[str, Any] | None = None
    login_url: str | None = None
    username_selector: str | None = None
    password_selector: str | None = None
    submit_selector: str | None = None
    wait_for_url: str | None = None
    wait_for_selector: str | None = None
    wait_timeout_ms: int = 15000


@dataclass
class ScrapeResult:
    """Result of a web scraping operation.

    Attributes:
        url: The scraped URL.
        html_content: Raw HTML content of the page.
        text_content: Extracted visible text content.
        markdown_content: Extracted main content as Markdown (LLM/RAG-ready).
        title: Page title.
        metadata: Extracted metadata (Open Graph, meta tags, etc.).
        structured_data: Embedded structured data (JSON-LD, microdata, OpenGraph...).
        links: List of discovered hyperlinks.
        images: List of discovered images.
        screenshot_path: File path to saved screenshot.
        screenshot_base64: Base64-encoded screenshot data.
        status: Operation completion status.
        error: Error message if operation failed.
        scraper_used: Name of the scraper that processed this.
        processing_time_seconds: Time taken to complete the scrape.
    """

    url: str
    html_content: str | None = None
    text_content: str | None = None
    markdown_content: str | None = None
    title: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    structured_data: dict[str, Any] = field(default_factory=dict)
    links: list[dict[str, Any]] = field(default_factory=list)
    images: list[dict[str, Any]] = field(default_factory=list)
    screenshot_path: str | None = None
    screenshot_base64: str | None = None
    status: ScrapeStatus = ScrapeStatus.COMPLETED
    error: str | None = None
    scraper_used: str | None = None
    processing_time_seconds: float = 0.0
    storage_state: dict[str, Any] | None = None


@dataclass
class ScrapeConfig:
    """Configuration for scraping operations.

    Attributes:
        delay: Delay between requests in seconds.
        timeout: Request timeout in seconds.
        max_retries: Maximum retry attempts on failure.
        follow_links: Whether to follow discovered links.
        max_depth: Maximum depth for link following.
        respect_robots_txt: Whether to respect robots.txt rules.
        screenshot: Whether to capture screenshots.
        user_agent: Custom User-Agent header string.
        headers: Custom HTTP headers to send.
        cookies: Custom cookies to send.
        javascript_enabled: Force a JS-rendering backend (Playwright) in AUTO mode.
        link_query: Optional topic used to score links and, when following
            links, to order the crawl frontier best-first.
        allowed_domains: When following links, only these domains (and their
            subdomains) are crawled. None means any same-site link.
        virtual_scroll: Optional scroll plan for lazily-loaded content
            (Playwright only).
        stealth: Apply browser-level stealth patches (Playwright only).
    """

    delay: float = 0.0
    timeout: float = 30.0
    max_retries: int = 3
    follow_links: bool = False
    max_depth: int = 1
    respect_robots_txt: bool = True
    screenshot: bool = False
    download_images: bool = False
    user_agent: str | None = None
    headers: dict[str, str] | None = None
    cookies: dict[str, str] | None = None
    javascript_enabled: bool = False
    link_query: str | None = None
    allowed_domains: list[str] | None = None
    virtual_scroll: Any = None
    stealth: bool = False
    auth_config: AuthConfig | None = None

    def __post_init__(self) -> None:
        """Initialize default mutable fields after dataclass creation."""
        if self.headers is None:
            self.headers = {}
        if self.cookies is None:
            self.cookies = {}


class BaseScraper(ABC):
    """Abstract base class for all scraper implementations.

    Defines the interface that all scrapers must implement and provides
    common utility methods for URL handling and validation.

    Attributes:
        config: Scraping configuration.
        name: Name of the scraper class.
    """

    def __init__(self, config: ScrapeConfig | None = None) -> None:
        """Initialize the scraper with configuration.

        Args:
            config: Scraping configuration. Uses defaults if not provided.
        """
        self.config = config or ScrapeConfig()
        self.name = self.__class__.__name__
        logger.info(
            f"{self.name} initialized with config: delay={self.config.delay}s, "
            f"timeout={self.config.timeout}s"
        )

    def _resolve_config(self, kwargs: dict[str, Any]) -> ScrapeConfig:
        """Return the request-local config, falling back to the instance default.

        ScraperManager keeps ONE shared scraper instance per type and passes the
        per-request ``ScrapeConfig`` as a ``config`` kwarg on every ``scrape``
        call. Reading from this resolved value (never ``self.config``) keeps a
        concurrent request from stomping this request's auth/cookies/headers on
        the shared instance.
        """
        return kwargs.get("config") or self.config

    @abstractmethod
    async def scrape(self, url: str, **kwargs: Any) -> ScrapeResult:
        """Scrape a URL and return the extracted content.

        Args:
            url: The URL to scrape.
            **kwargs: Additional scraper-specific options. ``config`` carries the
                per-request ``ScrapeConfig`` (see ``_resolve_config``).

        Returns:
            ScrapeResult with extracted content and metadata.
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """Clean up resources such as browsers and network connections."""
        pass

    def is_valid_url(self, url: str) -> bool:
        """Check if a URL is valid and has a scheme and host.

        Args:
            url: The URL to validate.

        Returns:
            True if the URL is valid, False otherwise.
        """
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False

    def normalize_url(self, url: str, base_url: str | None = None) -> str:
        """Normalize a URL by resolving relative paths and removing fragments.

        Args:
            url: The URL to normalize.
            base_url: Base URL for resolving relative URLs.

        Returns:
            Normalized absolute URL string.
        """
        try:
            url = url.split("#")[0]

            if base_url and not url.startswith(("http://", "https://")):
                url = urljoin(base_url, url)

            return url
        except Exception as e:
            logger.warning(f"Error normalizing URL {url}: {e}")
            return url

    def should_scrape_url(self, url: str, allowed_domains: list[str] | None = None) -> bool:
        """Check if a URL should be scraped based on domain restrictions.

        Args:
            url: The URL to check.
            allowed_domains: List of allowed domain names.

        Returns:
            True if the URL should be scraped, False otherwise.
        """
        if not self.is_valid_url(url):
            return False

        if allowed_domains:
            parsed = urlparse(url)
            domain = parsed.netloc

            if not any(
                domain == allowed or domain.endswith(f".{allowed}") for allowed in allowed_domains
            ):
                logger.debug(f"URL {url} not in allowed domains: {allowed_domains}")
                return False

        return True

    def extract_domain(self, url: str) -> str:
        """Extract the domain (netloc) from a URL.

        Args:
            url: The URL to extract the domain from.

        Returns:
            The domain string, or empty string on error.
        """
        try:
            parsed = urlparse(url)
            return parsed.netloc
        except Exception:
            return ""


__all__ = [
    "AuthConfig",
    "BaseScraper",
    "ScrapeResult",
    "ScrapeConfig",
    "ScrapeStatus",
    "ScraperType",
]
