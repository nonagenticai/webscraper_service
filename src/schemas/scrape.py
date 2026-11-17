"""Scrape request and response schemas.

This module defines Pydantic models for web scraping API requests
and responses, including single URL and bulk scraping operations.
"""

import re
from enum import StrEnum
from typing import Any, Literal

import soupsieve
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator, model_validator

from custom_types.constraint_types import SAFE_STRING_PATTERN

# Hard ceilings on scraped content so an unusually large page can't blow up
# response-model validation (→ 500, and for /bulk the whole batch). The router
# clamps scraped content to these at the boundary; the Field max_length is the
# matching backstop.
MAX_CONTENT_LENGTH = 10_000_000  # html_content / text_content (~10 MB)
MAX_SCREENSHOT_LENGTH = 50_000_000  # base64 screenshot (~50 MB)

# Per-field caps for discovered link/image attributes and the top-level URL/
# title/error strings. The router clamps each scraped value to these at the
# boundary, so an oversized value — e.g. a long inline ``data:`` URI kept as an
# image src, a srcset candidate list, or a tracking href — can't fail
# ScrapeResponse validation → 500 (and for /bulk, the whole batch). The Field
# max_length below is the matching backstop.
MAX_FIELD_LENGTH = 10_000  # link text/url, image src/alt/srcset, url/final_url/title/error
MAX_ATTR_LENGTH = 256  # link rel/type, image mime_type, scraper_used
MAX_IMAGE_TITLE_LENGTH = 1_000  # image title

# Declarative-extraction bounds. The schema is caller-supplied and walked
# recursively, so it is bounded on the REQUEST side (here) as well as during
# extraction (services.extraction_schema): an unbounded schema is a cheap way to
# make the service do unbounded work.
MAX_EXTRACTION_FIELDS = 100  # total fields across the whole schema tree
MAX_EXTRACTION_DEPTH = 5  # nesting levels of nested/list/nested_list
MAX_SELECTOR_LENGTH = 500

# Bare hostname: labels of alphanumerics/hyphens joined by dots. Deliberately
# does not accept schemes, ports, paths or wildcards — see the allowed_domains
# validator for why a natural-looking 'https://example.com/' has to be rejected.
_HOSTNAME_RE = re.compile(
    r"[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?(\.[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?)*"
)


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
    """

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class AuthType(StrEnum):
    """Authentication flow selector for a scrape request.

    Attributes:
        FORM: Fill a login form with username/password from the credential.
        STORAGE_STATE: Inject a previously captured Playwright storageState.
        BEARER: Set ``Authorization: Bearer <token>`` from the credential.
        BASIC: Set HTTP Basic auth on the browser context.
    """

    FORM = "form"
    STORAGE_STATE = "storage_state"
    BEARER = "bearer"
    BASIC = "basic"


class AuthConfig(BaseModel):
    """Reference to a stored credential plus flow hints for the scraper.

    The actual username/password/token lives in the secrets service and is
    fetched at scrape time using the caller's license token — it is NEVER
    passed inline in a scrape request.
    """

    type: AuthType = Field(default=AuthType.FORM, description="Authentication flow to apply")
    credential_id: str = Field(
        min_length=1,
        max_length=256,
        pattern=SAFE_STRING_PATTERN,
        description="ID of the credential stored in the secrets service",
    )
    login_url: HttpUrl | None = Field(
        default=None,
        description=(
            "Explicit login URL. If omitted, the scraper will attempt the login "
            "flow on whatever page the target URL redirects to."
        ),
    )
    username_selector: str | None = Field(
        default=None,
        max_length=1000,
        pattern=SAFE_STRING_PATTERN,
        description="CSS selector for the username/email input (optional, defaults to common patterns)",
    )
    password_selector: str | None = Field(
        default=None,
        max_length=1000,
        pattern=SAFE_STRING_PATTERN,
        description="CSS selector for the password input (optional, defaults to common patterns)",
    )
    submit_selector: str | None = Field(
        default=None,
        max_length=1000,
        pattern=SAFE_STRING_PATTERN,
        description="CSS selector for the submit button (optional, defaults to common patterns)",
    )
    wait_for_url: str | None = Field(
        default=None,
        max_length=1000,
        pattern=SAFE_STRING_PATTERN,
        description="URL regex/glob to wait for after submit (confirms login succeeded)",
    )
    wait_for_selector: str | None = Field(
        default=None,
        max_length=1000,
        pattern=SAFE_STRING_PATTERN,
        description="CSS selector to wait for after submit (alternative to wait_for_url)",
    )
    wait_timeout_ms: int = Field(
        default=15000,
        ge=1000,
        le=120000,
        description="Max milliseconds to wait for post-login state",
    )


class ExtractionFieldType(StrEnum):
    """How a schema field pulls its value out of the matched element.

    ``computed`` and ``regex`` from the upstream crawl4ai dialect are
    deliberately absent — see services.extraction_schema for why.
    """

    TEXT = "text"
    ATTRIBUTE = "attribute"
    HTML = "html"
    NESTED = "nested"
    LIST = "list"
    NESTED_LIST = "nested_list"


class ExtractionField(BaseModel):
    """One field in a declarative extraction schema.

    Attributes:
        name: Output key for this field in each extracted record.
        selector: CSS selector, relative to the enclosing element. Omit to read
            the enclosing element itself.
        type: How to pull the value (see ExtractionFieldType).
        attribute: Attribute name to read when ``type`` is ``attribute``.
        transform: Optional string transform applied to scalar values.
        default: Value used when the selector matches nothing.
        fields: Child fields, required for nested/list/nested_list.
    """

    model_config = ConfigDict(populate_by_name=True)

    name: str = Field(
        max_length=MAX_ATTR_LENGTH, pattern=SAFE_STRING_PATTERN, description="Output key"
    )
    selector: str | None = Field(
        default=None,
        max_length=MAX_SELECTOR_LENGTH,
        pattern=SAFE_STRING_PATTERN,
        description="CSS selector relative to the enclosing element",
    )
    type: ExtractionFieldType = Field(
        default=ExtractionFieldType.TEXT, description="How to extract the value"
    )
    attribute: str | None = Field(
        default=None,
        max_length=MAX_ATTR_LENGTH,
        pattern=SAFE_STRING_PATTERN,
        description="Attribute to read when type is 'attribute'",
    )
    transform: Literal["lowercase", "uppercase", "strip"] | None = Field(
        default=None, description="Optional transform for scalar values"
    )
    default: str | None = Field(
        default=None,
        max_length=MAX_FIELD_LENGTH,
        description="Value used when the selector matches nothing",
    )
    fields: list[ExtractionField] = Field(
        default_factory=list, description="Child fields for nested/list/nested_list"
    )

    @model_validator(mode="after")
    def _check_type_requirements(self) -> ExtractionField:
        """Reject shapes that would silently extract nothing.

        A container field with no children, or an ``attribute`` field with no
        attribute name, is always a caller mistake. Failing at validation (422)
        beats returning an empty ``extracted_data`` that reads as "the page had
        no matching rows".
        """
        container = self.type in (
            ExtractionFieldType.NESTED,
            ExtractionFieldType.LIST,
            ExtractionFieldType.NESTED_LIST,
        )
        if container:
            if not self.fields:
                raise ValueError(f"field {self.name!r} of type {self.type} requires 'fields'")
            if not self.selector:
                raise ValueError(f"field {self.name!r} of type {self.type} requires 'selector'")
        elif self.fields:
            raise ValueError(f"field {self.name!r} of type {self.type} must not declare 'fields'")

        if self.type is ExtractionFieldType.ATTRIBUTE and not self.attribute:
            raise ValueError(f"field {self.name!r} of type 'attribute' requires 'attribute'")
        return self


def _count_fields(fields: list[ExtractionField], depth: int = 1) -> tuple[int, int]:
    """Return (total field count, maximum nesting depth) for a schema tree."""
    total = len(fields)
    max_depth = depth
    for field in fields:
        if field.fields:
            child_total, child_depth = _count_fields(field.fields, depth + 1)
            total += child_total
            max_depth = max(max_depth, child_depth)
    return total, max_depth


class ExtractionSchema(BaseModel):
    """Declarative CSS schema describing repeated records on a page.

    Wire-compatible with crawl4ai's ``JsonCssExtractionStrategy`` dialect:
    ``baseSelector`` is accepted as an alias for ``base_selector``, so an
    existing crawl4ai schema can be sent verbatim.

    Attributes:
        name: Human label for the schema (echoed nowhere; aids debugging).
        base_selector: Selector matching one element per record.
        fields: Fields to extract from each matched record.
    """

    model_config = ConfigDict(populate_by_name=True)

    name: str = Field(
        default="extraction",
        max_length=MAX_ATTR_LENGTH,
        pattern=SAFE_STRING_PATTERN,
        description="Human label for this schema",
    )
    base_selector: str = Field(
        alias="baseSelector",
        max_length=MAX_SELECTOR_LENGTH,
        pattern=SAFE_STRING_PATTERN,
        description="CSS selector matching one element per record",
    )
    fields: list[ExtractionField] = Field(
        min_length=1, description="Fields to extract from each matched record"
    )

    @model_validator(mode="after")
    def _check_size(self) -> ExtractionSchema:
        """Bound the schema tree so a request cannot ask for unbounded work."""
        total, depth = _count_fields(self.fields)
        if total > MAX_EXTRACTION_FIELDS:
            raise ValueError(
                f"extraction schema has {total} fields, limit is {MAX_EXTRACTION_FIELDS}"
            )
        if depth > MAX_EXTRACTION_DEPTH:
            raise ValueError(
                f"extraction schema nests {depth} levels, limit is {MAX_EXTRACTION_DEPTH}"
            )
        _assert_selectors_compile(self.base_selector, self.fields)
        return self


def _assert_selectors_compile(base_selector: str, fields: list[ExtractionField]) -> None:
    """Reject a schema whose CSS does not parse, before anything is scraped.

    An unparseable selector used to be caught during extraction and downgraded
    to an empty ``extracted_data`` — indistinguishable from "the page had no
    matching rows", after a full scrape had already been paid for. soupsieve is
    the same engine bs4's ``.select()`` uses, so compiling here is exactly the
    check that would have failed later.
    """
    try:
        soupsieve.compile(base_selector)
    except Exception as e:
        raise ValueError(f"baseSelector {base_selector!r} is not a valid CSS selector: {e}") from e

    stack = list(fields)
    while stack:
        field = stack.pop()
        if field.selector:
            try:
                soupsieve.compile(field.selector)
            except Exception as e:
                raise ValueError(
                    f"field {field.name!r} selector {field.selector!r} "
                    f"is not a valid CSS selector: {e}"
                ) from e
        stack.extend(field.fields)


class VirtualScrollConfig(BaseModel):
    """How to scroll a page to materialize lazily-loaded content.

    Playwright-only: the other backends do not execute JavaScript, so supplying
    this forces the Playwright backend in AUTO mode.

    Attributes:
        container_selector: Element to scroll. Omit to scroll the window.
            Required to recover *virtualized* lists that recycle DOM nodes —
            without a container there is nothing to accumulate children from.
        scroll_count: Maximum scroll steps. Bounded because each step costs a
            real wait against the caller's timeout.
        scroll_by: Pixels per step; 0 means one viewport/container height.
        wait_ms: Pause after each step, for content to load.
    """

    container_selector: str | None = Field(
        default=None,
        max_length=MAX_SELECTOR_LENGTH,
        pattern=SAFE_STRING_PATTERN,
        description="Element to scroll; omit to scroll the window",
    )
    scroll_count: int = Field(default=10, ge=1, le=30, description="Maximum number of scroll steps")
    scroll_by: int = Field(
        default=0,
        ge=0,
        le=20_000,
        description="Pixels to scroll per step; 0 means one viewport height",
    )
    wait_ms: int = Field(
        default=500, ge=50, le=5_000, description="Pause after each step, in milliseconds"
    )


#: Backends that execute JavaScript, and therefore can honour virtual_scroll
#: and stealth. AUTO is included because it *routes* to Playwright when either
#: option is present (see ScraperManager._select_auto_scraper).
_BROWSER_BACKENDS = frozenset({ScraperType.PLAYWRIGHT, ScraperType.AUTO})


def _assert_browser_backend(
    scraper_type: ScraperType, virtual_scroll: object | None, stealth: bool
) -> None:
    """Raise if browser-only options were sent to a non-browser backend."""
    if scraper_type in _BROWSER_BACKENDS:
        return
    requested = [
        name
        for name, present in (("virtual_scroll", virtual_scroll is not None), ("stealth", stealth))
        if present
    ]
    if requested:
        raise ValueError(
            f"{' and '.join(requested)} require a JavaScript-capable backend; "
            f"scraper_type={scraper_type.value!r} cannot honour them. "
            "Use 'playwright' or 'auto'."
        )


class ScrapeRequest(BaseModel):
    """Request model for scraping a single URL.

    Attributes:
        url: The URL to scrape.
        scraper_type: Backend scraper to use.
        javascript_enabled: Force Playwright in AUTO mode (ignored otherwise).
        timeout: Request timeout in seconds.
        wait_for_selector: CSS selector to wait for before scraping.
        wait_until: Page load wait condition.
        screenshot: Whether to capture a screenshot.
        viewport_width: Browser viewport width in pixels.
        viewport_height: Browser viewport height in pixels.
        user_agent: Custom User-Agent header.
        headers: Custom HTTP headers.
        cookies: Custom cookies to send.
        link_query: Optional topic used to score discovered links.
        sort_links_by_score: Whether to return links ranked by score.
    """

    url: HttpUrl = Field(description="URL to scrape")
    scraper_type: ScraperType = Field(
        default=ScraperType.AUTO,
        description="Type of scraper to use (playwright for JS-heavy sites, scrapy for bulk)",
    )
    javascript_enabled: bool = Field(
        default=False,
        description=(
            "Force a JS-rendering backend (Playwright) when scraper_type is 'auto'. "
            "Ignored when scraper_type names an explicit backend."
        ),
    )
    no_cache: bool = Field(
        default=False,
        description=(
            "Bypass the result cache for this request (always fetch fresh and do "
            "not store). The cache only ever holds anonymous COMPLETED scrapes."
        ),
    )
    timeout: float = Field(default=30.0, ge=1.0, le=300.0, description="Request timeout in seconds")
    delay: float = Field(
        default=0.0,
        ge=0.0,
        le=30.0,
        description="Post-load delay in seconds before content extraction",
    )
    wait_for_selector: str | None = Field(
        default=None,
        max_length=1000,
        pattern=SAFE_STRING_PATTERN,
        description="CSS selector to wait for before scraping (Playwright only)",
    )
    wait_until: Literal["load", "domcontentloaded", "networkidle", "commit"] = Field(
        default="load",
        description=(
            "Page-load wait condition: 'load' (default), 'domcontentloaded', "
            "'networkidle', or 'commit'. 'networkidle' is an explicit opt-in only — "
            "on live SPAs with analytics/websockets/polling the network never goes "
            "500ms-quiet, so it blocks the whole timeout and returns empty content "
            "on a good page. An invalid value is rejected with 422 (rather than "
            "reaching page.goto and returning a FAILED result)."
        ),
    )
    screenshot: bool = Field(default=False, description="Capture screenshot of page")
    download_images: bool = Field(
        default=False,
        description="Download image content and include base64-encoded bytes in response",
    )
    viewport_width: int = Field(default=1920, ge=320, le=3840, description="Viewport width")
    viewport_height: int = Field(default=1080, ge=240, le=2160, description="Viewport height")
    user_agent: str | None = Field(
        default=None,
        max_length=1000,
        pattern=SAFE_STRING_PATTERN,
        description="Custom user agent string",
    )
    headers: dict[str, str] | None = Field(default=None, description="Custom HTTP headers")
    cookies: dict[str, str] | None = Field(default=None, description="Custom cookies")
    link_query: str | None = Field(
        default=None,
        max_length=500,
        pattern=SAFE_STRING_PATTERN,
        description=(
            "Optional topic (e.g. 'pricing plans') used to score discovered links. "
            "Every link always carries a score; supplying a query blends topical "
            "relevance into it. Does not filter or reorder unless "
            "sort_links_by_score is set."
        ),
    )
    sort_links_by_score: bool = Field(
        default=False,
        description=(
            "Return links[] ranked by score (descending) instead of document order. "
            "Off by default because document order is the existing contract."
        ),
    )
    extraction_schema: ExtractionSchema | None = Field(
        default=None,
        description=(
            "Optional declarative CSS schema for pulling repeated records "
            "(listings, tables, team pages) into extracted_data. Runs against the "
            "page HTML, so it works with every backend. Wire-compatible with "
            "crawl4ai's JsonCssExtractionStrategy dialect."
        ),
    )
    virtual_scroll: VirtualScrollConfig | None = Field(
        default=None,
        description=(
            "Scroll the page (or a container) before extracting, so infinite "
            "feeds yield more than the first viewport. Forces the Playwright "
            "backend, since the other backends do not run JavaScript."
        ),
    )
    stealth: bool = Field(
        default=False,
        description=(
            "Apply browser-level stealth patches (navigator.webdriver and "
            "friends) to the Playwright context. Applied automatically when "
            "AUTO escalates to Playwright after curl_cffi was already blocked; "
            "set explicitly to force it. Requires a browser backend: rejected "
            "with a 422 alongside an explicit non-browser scraper_type."
        ),
    )
    auth_config: AuthConfig | None = Field(
        default=None,
        description=(
            "Optional authentication configuration referencing a stored credential. "
            "When present, forces the Playwright scraper for form/storage_state types."
        ),
    )

    @model_validator(mode="after")
    def _check_browser_only_options(self) -> ScrapeRequest:
        """Reject browser-only options aimed at a backend that cannot run them.

        AUTO routes ``virtual_scroll``/``stealth`` to Playwright, but naming
        ``httpx``/``curl_cffi``/``scrapy`` explicitly used to be accepted and
        then quietly ignored: the caller got a first-viewport, un-stealthed
        result with ``status: completed`` and no indication their option had
        been dropped. Fail the request instead.
        """
        _assert_browser_backend(self.scraper_type, self.virtual_scroll, self.stealth)
        return self

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "url": "https://example.com",
                "scraper_type": "auto",
                "timeout": 30.0,
                "screenshot": False,
                "viewport_width": 1920,
                "viewport_height": 1080,
            }
        }
    )


class LinkInfo(BaseModel):
    """Information about a discovered hyperlink.

    Attributes:
        text: The link's anchor text.
        url: The link's target URL.
        rel: The link's rel attribute value.
        type: Classification as 'internal' or 'external'.
        score: Relevance score in [0, 1] (see services.link_scoring). Present
            on every link; higher is more relevant. When the request carries a
            ``link_query`` the score blends intrinsic link quality with how well
            the anchor text and URL slug match that query. ``type == "internal"``
            remains the same-domain signal — it is not duplicated here.
    """

    text: str = Field(
        max_length=MAX_FIELD_LENGTH, pattern=SAFE_STRING_PATTERN, description="Link text"
    )
    url: str = Field(
        max_length=MAX_FIELD_LENGTH, pattern=SAFE_STRING_PATTERN, description="Link URL"
    )
    rel: str | None = Field(
        default=None,
        max_length=MAX_ATTR_LENGTH,
        pattern=SAFE_STRING_PATTERN,
        description="Link rel attribute",
    )
    type: str | None = Field(
        default=None,
        max_length=MAX_ATTR_LENGTH,
        pattern=SAFE_STRING_PATTERN,
        description="internal or external",
    )
    score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Relevance score in [0, 1]; higher is more relevant",
    )


class ImageInfo(BaseModel):
    """Information about a discovered image.

    Attributes:
        src: The image source URL.
        alt: The image alt text.
        title: The image title attribute.
        width: Natural width in pixels.
        height: Natural height in pixels.
        content_base64: Base64-encoded image bytes (when download_images=true).
        mime_type: Detected MIME type of downloaded image.
        byte_size: Size of downloaded image in bytes.
    """

    src: str = Field(
        max_length=MAX_FIELD_LENGTH, pattern=SAFE_STRING_PATTERN, description="Image source URL"
    )
    alt: str = Field(
        default="",
        max_length=MAX_FIELD_LENGTH,
        pattern=SAFE_STRING_PATTERN,
        description="Alt text",
    )
    title: str = Field(
        default="",
        max_length=MAX_IMAGE_TITLE_LENGTH,
        pattern=SAFE_STRING_PATTERN,
        description="Title attribute",
    )
    width: int | None = Field(default=None, description="Natural width")
    height: int | None = Field(default=None, description="Natural height")
    srcset: str | None = Field(
        default=None,
        max_length=MAX_FIELD_LENGTH,
        description="srcset attribute for responsive images",
    )
    content_base64: str | None = Field(default=None, description="Base64-encoded image content")
    mime_type: str | None = Field(
        default=None,
        max_length=MAX_ATTR_LENGTH,
        description="MIME type of the image",
    )
    byte_size: int | None = Field(default=None, description="Size of image in bytes")


class ScrapeResponse(BaseModel):
    """Response model for a completed scrape operation.

    Attributes:
        url: The original requested URL.
        final_url: The final URL after redirects.
        html_content: Raw HTML content of the page.
        text_content: Extracted visible text content.
        markdown_content: Extracted main content as Markdown (LLM/RAG-ready).
        title: Page title.
        metadata: Extracted metadata (Open Graph, meta tags, etc.).
        structured_data: Embedded structured data (JSON-LD, microdata, OpenGraph, microformats).
        links: List of discovered hyperlinks.
        images: List of discovered images.
        screenshot_base64: Base64-encoded screenshot PNG.
        status: Operation status.
        error: Error message if status is FAILED.
        scraper_used: Name of the scraper that processed the request.
        processing_time_seconds: Time taken to complete the scrape.
    """

    url: str = Field(
        max_length=MAX_FIELD_LENGTH, pattern=SAFE_STRING_PATTERN, description="Scraped URL"
    )
    final_url: str | None = Field(
        default=None,
        max_length=MAX_FIELD_LENGTH,
        pattern=SAFE_STRING_PATTERN,
        description="Final URL after redirects",
    )
    html_content: str | None = Field(
        default=None, max_length=MAX_CONTENT_LENGTH, description="Raw HTML content"
    )
    text_content: str | None = Field(
        default=None, max_length=MAX_CONTENT_LENGTH, description="Extracted text content"
    )
    markdown_content: str | None = Field(
        default=None,
        max_length=MAX_CONTENT_LENGTH,
        description="Extracted main content as Markdown (LLM/RAG-ready)",
    )
    title: str | None = Field(
        default=None,
        max_length=MAX_FIELD_LENGTH,
        pattern=SAFE_STRING_PATTERN,
        description="Page title",
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Page metadata")
    structured_data: dict[str, Any] = Field(
        default_factory=dict,
        description="Embedded structured data (JSON-LD, microdata, OpenGraph, microformats)",
    )
    extracted_data: list[dict[str, Any]] | None = Field(
        default=None,
        description=(
            "Records pulled by the request's extraction_schema, one dict per "
            "matched element. None when no schema was supplied; an empty list "
            "means a schema ran and matched nothing."
        ),
    )
    links: list[LinkInfo] = Field(default_factory=list, description="Discovered links")
    images: list[ImageInfo] = Field(default_factory=list, description="Discovered images")
    screenshot_base64: str | None = Field(
        default=None, max_length=MAX_SCREENSHOT_LENGTH, description="Base64 encoded screenshot"
    )
    status: ScrapeStatus = Field(description="Scrape operation status")
    error: str | None = Field(
        default=None,
        max_length=MAX_FIELD_LENGTH,
        pattern=SAFE_STRING_PATTERN,
        description="Error message if failed",
    )
    scraper_used: str = Field(
        max_length=MAX_ATTR_LENGTH,
        pattern=SAFE_STRING_PATTERN,
        description="Name of scraper that processed this",
    )
    processing_time_seconds: float = Field(description="Time taken to scrape")
    storage_state: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Playwright storageState captured after successful auth. Cache and "
            "reuse via auth_config.type='storage_state' to avoid re-logging-in."
        ),
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "url": "https://example.com",
                "final_url": "https://example.com/",
                "title": "Example Domain",
                "text_content": "Example Domain\nThis domain is for use in examples...",
                "status": "completed",
                "scraper_used": "PlaywrightScraper",
                "processing_time_seconds": 2.5,
            }
        }
    )


class BulkScrapeRequest(BaseModel):
    """Request model for scraping multiple URLs.

    Attributes:
        urls: List of URLs to scrape.
        scraper_type: Backend scraper to use for all URLs.
        javascript_enabled: Force Playwright in AUTO mode (ignored otherwise).
        timeout: Request timeout in seconds per URL.
        follow_links: Whether to follow links on scraped pages.
        max_depth: Maximum crawl depth when following links.
        allowed_domains: Domains allowed when following links.
        link_query: Optional topic used to score links and order the crawl frontier.
        sort_links_by_score: Whether to return links ranked by score.
    """

    auth_config: AuthConfig | None = Field(
        default=None,
        description=(
            "Optional authentication configuration referencing a stored credential. "
            "Login is performed once and the session is reused across all URLs."
        ),
    )
    urls: list[HttpUrl] = Field(description="List of URLs to scrape", min_length=1, max_length=100)
    scraper_type: ScraperType = Field(
        default=ScraperType.AUTO,
        description="Type of scraper to use",
    )
    javascript_enabled: bool = Field(
        default=False,
        description=(
            "Force a JS-rendering backend (Playwright) when scraper_type is 'auto'. "
            "Ignored when scraper_type names an explicit backend."
        ),
    )
    timeout: float = Field(default=30.0, ge=1.0, le=300.0, description="Request timeout in seconds")
    delay: float = Field(
        default=0.0,
        ge=0.0,
        le=30.0,
        description="Post-load delay in seconds before content extraction",
    )
    wait_for_selector: str | None = Field(
        default=None,
        max_length=1000,
        pattern=SAFE_STRING_PATTERN,
        description="CSS selector to wait for before scraping (Playwright only)",
    )
    wait_until: Literal["load", "domcontentloaded", "networkidle", "commit"] = Field(
        default="load",
        description=(
            "Page-load wait condition: 'load' (default), 'domcontentloaded', "
            "'networkidle', or 'commit'. 'networkidle' is an explicit opt-in only — "
            "on live SPAs with analytics/websockets/polling the network never goes "
            "500ms-quiet, so it blocks the whole timeout and returns empty content "
            "on a good page. An invalid value is rejected with 422 (rather than "
            "reaching page.goto and returning a FAILED result)."
        ),
    )
    download_images: bool = Field(
        default=False,
        description="Download image content and include base64-encoded bytes in response",
    )
    screenshot: bool = Field(default=False, description="Capture screenshot of each page")
    follow_links: bool = Field(default=False, description="Follow links on scraped pages")
    max_depth: int = Field(default=1, ge=1, le=5, description="Maximum crawl depth")
    link_query: str | None = Field(
        default=None,
        max_length=500,
        pattern=SAFE_STRING_PATTERN,
        description=(
            "Optional topic (e.g. 'pricing plans') used to score discovered links. "
            "With follow_links it also orders the Scrapy crawl frontier best-first, "
            "so a depth-limited crawl spends its budget on relevant pages."
        ),
    )
    sort_links_by_score: bool = Field(
        default=False,
        description="Return links[] ranked by score (descending) instead of document order",
    )
    extraction_schema: ExtractionSchema | None = Field(
        default=None,
        description=(
            "Optional declarative CSS schema applied to every scraped page, "
            "producing extracted_data per result."
        ),
    )
    allowed_domains: list[str] | None = Field(
        default=None,
        max_length=100,
        description=(
            "Restrict link following to these domains (subdomains included). "
            "Applied to every followed link; the seed URLs themselves are always "
            "scraped. Omit to follow any same-site link the page declares. "
            "Entries are bare hostnames — 'example.com', not a URL or a wildcard."
        ),
    )

    @field_validator("allowed_domains")
    @classmethod
    def _check_allowed_domains(cls, value: list[str] | None) -> list[str] | None:
        """Reject entries that can never match, instead of crawling nothing.

        The filter compares against a URL's bare host, so a caller who passes
        ``https://example.com/`` or ``*.example.com`` — both natural guesses —
        matches no link at all. That produced an empty crawl reported as a
        successful one; it is now a 422 that says which entry is wrong.
        """
        if not value:
            return value
        for entry in value:
            if entry != entry.strip():
                raise ValueError(f"allowed_domains entry {entry!r} must not carry whitespace")
            host = entry.lower()
            if not host:
                raise ValueError("allowed_domains entries must not be empty")
            if "/" in host or ":" in host or "*" in host:
                raise ValueError(
                    f"allowed_domains entry {entry!r} must be a bare hostname "
                    "(e.g. 'example.com'), not a URL, port or wildcard"
                )
            if not _HOSTNAME_RE.fullmatch(host):
                raise ValueError(f"allowed_domains entry {entry!r} is not a valid hostname")
        return value

    virtual_scroll: VirtualScrollConfig | None = Field(
        default=None,
        description=(
            "Scroll each page (or a container) before extracting. Forces the "
            "Playwright backend; not combinable with follow_links, which crawls "
            "via Scrapy."
        ),
    )
    stealth: bool = Field(
        default=False,
        description=(
            "Apply browser-level stealth patches to each page. Requires a "
            "browser backend; not combinable with follow_links."
        ),
    )

    @model_validator(mode="after")
    def _check_browser_only_options(self) -> BulkScrapeRequest:
        """Reject browser-only options the chosen crawl mode cannot honour.

        Two ways this used to fail silently, both ending in a 200 that looked
        complete: an explicit non-browser ``scraper_type``, and ``follow_links``
        — which takes precedence in AUTO routing and sends the whole batch to
        Scrapy, where ``virtual_scroll`` and ``stealth`` mean nothing.
        """
        _assert_browser_backend(self.scraper_type, self.virtual_scroll, self.stealth)
        if self.follow_links or self.max_depth > 1:
            requested = [
                name
                for name, present in (
                    ("virtual_scroll", self.virtual_scroll is not None),
                    ("stealth", self.stealth),
                )
                if present
            ]
            if requested:
                raise ValueError(
                    f"{' and '.join(requested)} cannot be combined with follow_links/"
                    "max_depth: crawling runs on the Scrapy backend, which does not "
                    "execute JavaScript. Crawl first, then scrape the pages you want "
                    "with the browser backend."
                )
        return self

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "urls": ["https://example.com", "https://example.org"],
                "scraper_type": "scrapy",
                "timeout": 30.0,
                "follow_links": False,
            }
        }
    )


class BulkScrapeResponse(BaseModel):
    """Response model for bulk scrape operations.

    Attributes:
        total_urls: Number of URLs in the request.
        completed: Number of successfully scraped URLs.
        failed: Number of URLs that failed to scrape.
        results: Individual scrape results for each URL.
        processing_time_seconds: Total time for all URLs.
    """

    total_urls: int = Field(description="Total URLs requested")
    completed: int = Field(description="Successfully scraped URLs")
    failed: int = Field(description="Failed URLs")
    results: list[ScrapeResponse] = Field(description="Individual scrape results")
    processing_time_seconds: float = Field(description="Total processing time")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "total_urls": 2,
                "completed": 2,
                "failed": 0,
                "results": [],
                "processing_time_seconds": 5.0,
            }
        }
    )
