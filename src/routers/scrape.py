"""Scrape router - API endpoints for web scraping operations.

This module provides REST API endpoints for single URL and bulk
web scraping operations using Playwright or Scrapy backends.
"""

import asyncio
import logging
import time
from typing import Any

from fastapi import APIRouter, HTTPException, Request, Response, status

from config import get_settings
from custom_types.constraint_types import sanitize_safe_string as _clean
from routers.responses import SCRAPE_RESPONSES
from schemas.scrape import (
    MAX_ATTR_LENGTH,
    MAX_CONTENT_LENGTH,
    MAX_FIELD_LENGTH,
    MAX_IMAGE_TITLE_LENGTH,
    MAX_SCREENSHOT_LENGTH,
    BulkScrapeRequest,
    BulkScrapeResponse,
    ExtractionSchema,
    ImageInfo,
    LinkInfo,
    ScrapeRequest,
    ScrapeResponse,
    ScraperType,
    ScrapeStatus,
)
from schemas.scrape import (
    AuthConfig as SchemaAuthConfig,
)
from services.base_scraper import AuthConfig as ServiceAuthConfig
from services.base_scraper import ScrapeConfig
from services.base_scraper import ScraperType as ServiceScraperType
from services.base_scraper import ScrapeStatus as ServiceScrapeStatus
from services.cache import get_result_cache
from services.extraction_schema import extract_structured
from services.link_scoring import LinkScorer
from services.metrics import record_cache
from services.scraper_manager import get_scraper_manager
from services.secrets_client import (
    CredentialNotFoundError,
    SecretsClient,
    SecretsServiceError,
    UrlCredential,
)
from services.url_guard import UrlNotAllowedError, assert_allowed_url

logger = logging.getLogger(__name__)

# Per-field string lengths are clamped, but the NUMBER of links/images a page can
# yield is otherwise unbounded — a body of pure <a>/<img> tags produces a
# correspondingly huge pydantic-validated response. Cap the count too (generous
# headroom for legitimate pages); the body-size cap upstream already bounds the
# common case, this is cheap defense-in-depth.
MAX_LINKS = 5000
MAX_IMAGES = 2000
# Ceiling on links we will SCORE. Scoring has to precede the MAX_LINKS cap for
# ranking to mean anything, which makes the per-link work the thing that needs
# bounding — and it is synchronous CPU on the event loop, so the bound is a
# latency budget for every OTHER request in flight, not just this one. 10x
# MAX_LINKS admitted ~50k scorings per page; 2x still exceeds any real page by
# two orders of magnitude (the median page carries well under 200 links).
MAX_SCORED_LINKS = MAX_LINKS * 2


def _clean_opt(value: str | None) -> str | None:
    """Sanitize an optional scraped string (None passes through)."""
    return _clean(value) if value is not None else None


def _clamp_opt(value: str | None, max_len: int) -> str | None:
    """Truncate oversized scraped content to its schema cap (None passes through).

    An unusually large page (>~10 MB text/html, >~50 MB screenshot) would
    otherwise fail ScrapeResponse validation → caught by the route's
    ``except Exception`` → HTTP 500 (and for /bulk, the WHOLE batch). A clamped
    200 is strictly better than a 500. These fields are length-only constrained
    (no pattern), so truncation is safe and never corrupts the value.
    """
    if value is not None and len(value) > max_len:
        return value[:max_len]
    return value


def _clean_clamp(value: str, max_len: int) -> str:
    """Sanitize then truncate a scraped string field to its schema cap.

    The pattern-constrained per-link/per-image fields (and the top-level
    url/title/error) are length-bounded too. ``_clean`` only strips control
    chars — it never truncates — so an oversized scraped value (a long inline
    ``data:`` URI kept as an image src, a srcset candidate list, a tracking
    href) would still fail validation → 500 (and for /bulk the whole batch).
    Clamp at the boundary: a shape-stable truncated 200 beats a 500.
    """
    cleaned = _clean(value)
    return cleaned[:max_len] if len(cleaned) > max_len else cleaned


def _clean_clamp_opt(value: str | None, max_len: int) -> str | None:
    """``_clean_clamp`` for an optional field (None passes through)."""
    return _clean_clamp(value, max_len) if value is not None else None


def _sanitize_metadata(metadata: dict[str, object]) -> dict[str, object]:
    """Strip control chars from str values of scraped metadata at the boundary.

    ``metadata`` is ``dict[str, Any]`` (no pattern) so it can't 500, but its
    values come from arbitrary ``<meta>`` content and could carry control chars
    into downstream logs/consumers — every other returned string field is
    boundary-cleaned, so clean these too. Non-str values (ints, lists, nested
    dicts) are left untouched.
    """
    return {k: _clean(v) if isinstance(v, str) else v for k, v in metadata.items()}


router = APIRouter(prefix="/api/scrape", tags=["Scrape"])


def _get_secrets_client() -> SecretsClient:
    settings = get_settings()
    return SecretsClient(
        base_url=settings.secrets_service_url,
        timeout=settings.secrets_service_timeout,
    )


def _build_service_auth(
    schema_auth: SchemaAuthConfig, credential: UrlCredential
) -> ServiceAuthConfig:
    """Merge request-layer AuthConfig (flow hints) with fetched credential payload.

    Raises HTTPException(400) if required credential fields are missing for the
    requested auth type.
    """
    creds = credential.credentials
    auth_type = schema_auth.type.value

    if auth_type == "form":
        missing = [k for k in ("username", "password") if not creds.get(k)]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"credential {schema_auth.credential_id} missing field "
                    f"{missing[0]!r} for auth type 'form'"
                ),
            )
        username, password = creds["username"], creds["password"]
        token = ""
        storage_state = None
    elif auth_type == "basic":
        missing = [k for k in ("username", "password") if not creds.get(k)]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"credential {schema_auth.credential_id} missing field "
                    f"{missing[0]!r} for auth type 'basic'"
                ),
            )
        username, password = creds["username"], creds["password"]
        token = ""
        storage_state = None
    elif auth_type == "bearer":
        if not creds.get("auth_token"):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"credential {schema_auth.credential_id} missing field "
                    f"'auth_token' for auth type 'bearer'"
                ),
            )
        username, password = "", ""
        token = creds["auth_token"]
        storage_state = None
    elif auth_type == "storage_state":
        if not creds.get("storage_state"):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"credential {schema_auth.credential_id} missing field "
                    f"'storage_state' for auth type 'storage_state'"
                ),
            )
        username, password, token = "", "", ""
        storage_state = creds["storage_state"]
    else:  # pragma: no cover — StrEnum prevents this
        raise HTTPException(status_code=400, detail=f"unsupported auth type {auth_type!r}")

    return ServiceAuthConfig(
        type=auth_type,
        username=username,
        password=password,
        token=token,
        storage_state=storage_state,
        login_url=str(schema_auth.login_url) if schema_auth.login_url else None,
        username_selector=schema_auth.username_selector,
        password_selector=schema_auth.password_selector,
        submit_selector=schema_auth.submit_selector,
        wait_for_url=schema_auth.wait_for_url,
        wait_for_selector=schema_auth.wait_for_selector,
        wait_timeout_ms=schema_auth.wait_timeout_ms,
    )


async def _resolve_auth_config(
    schema_auth: SchemaAuthConfig | None, request: Request
) -> ServiceAuthConfig | None:
    """Fetch the referenced credential from the secrets service and build the service-layer AuthConfig.

    Returns None when ``schema_auth`` is None.
    Raises HTTPException(404) for CredentialNotFoundError, 502 for other secrets-service errors.
    """
    if schema_auth is None:
        return None

    license_token: str | None = getattr(request.state, "license_token", None)
    tenant_id: str = getattr(request.state, "tenant_id", "") or ""

    client = _get_secrets_client()
    try:
        credential = await client.get_url_credential(
            schema_auth.credential_id,
            license_token=license_token,
            tenant_id=tenant_id,
        )
    except CredentialNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except SecretsServiceError as e:
        raise HTTPException(
            status_code=502,
            detail=f"secrets service unavailable: {e}",
        ) from e

    return _build_service_auth(schema_auth, credential)


async def _guard_url(url: str, *, field: str = "url") -> None:
    """SSRF-validate a caller-supplied URL, mapping rejection to HTTP 400.

    Used for the single-scrape target and any auth ``login_url``; the bulk
    endpoint validates per-URL inside the manager (→ FAILED result) so one
    blocked URL doesn't fail the whole batch. Resolution runs off the event
    loop (it does blocking DNS).
    """
    settings = get_settings()
    if not settings.ssrf_protection_enabled:
        return
    try:
        await asyncio.to_thread(assert_allowed_url, url, settings.ssrf_allowed_hosts)
    except UrlNotAllowedError as e:
        raise HTTPException(status_code=400, detail=f"{field}: {e}") from e


def convert_scraper_type(schema_type: ScraperType) -> ServiceScraperType:
    """Convert API schema ScraperType to service layer ScraperType.

    Args:
        schema_type: The scraper type from the API schema.

    Returns:
        Corresponding service layer ScraperType enum value.
    """
    return ServiceScraperType(schema_type.value)


def convert_scrape_status(service_status: ServiceScrapeStatus) -> ScrapeStatus:
    """Convert service layer ScrapeStatus to API schema ScrapeStatus.

    Args:
        service_status: The scrape status from the service layer.

    Returns:
        Corresponding API schema ScrapeStatus enum value.
    """
    return ScrapeStatus(service_status.value)


def _run_extraction(
    html: str | None, schema: ExtractionSchema | None
) -> tuple[list[dict[str, Any]] | None, bool, list[str]]:
    """Apply the request's extraction schema to the page HTML, if any.

    Runs against the UNCLAMPED html so a page just over MAX_CONTENT_LENGTH still
    extracts fully — the clamp exists to bound the response body, not the work.

    Returns:
        ``(extracted_data, truncated, field_errors)``. ``extracted_data`` is
        None ONLY when no schema was supplied; a schema against an empty page
        returns ``[]``, so "not asked for" stays distinguishable from "asked
        for, found nothing" even when the backend produced no HTML.
        ``truncated`` is True when a cap bit, so the caller is not handed a
        short list as a complete one. ``field_errors`` names fields whose
        (already-valid) selector raised while actually running against this
        page — distinct from a selector that correctly matched nothing, which
        is a normal result and never appears here.
    """
    if schema is None:
        return None, False, []
    if not html:
        return [], False, []
    result = extract_structured(html, schema)
    return result.items, result.truncated, result.field_errors


def _convert_links(
    raw_links: list[dict],
    link_query: str | None = None,
    sort_by_score: bool = False,
) -> list[LinkInfo]:
    """Convert raw link dictionaries to LinkInfo models, scoring each one.

    Scoring runs over the raw list BEFORE the ``MAX_LINKS`` cap: truncating
    first would discard the highest-scoring links on a link-heavy page and make
    the ranking silently meaningless. That deliberately moved the cap after the
    per-link work, so the work itself is bounded by ``MAX_SCORED_LINKS`` — a
    page carrying half a million anchors must not turn into half a million
    model validations. Only the surviving links become ``LinkInfo`` models.

    Sorting is opt-in — document order is the established contract for existing
    consumers, and the ``{text, url, rel, type}`` shape is unchanged.

    Args:
        raw_links: List of dictionaries with link data.
        link_query: Optional topic to blend into the relevance score.
        sort_by_score: Return links ranked by score instead of document order.

    Returns:
        List of LinkInfo Pydantic models, capped at MAX_LINKS.
    """
    scorer = LinkScorer.for_query(link_query)
    scored: list[tuple[float, dict]] = [
        (
            scorer.score(
                url=str(link.get("url") or ""),
                text=str(link.get("text") or ""),
                rel=link.get("rel"),
                link_type=link.get("type"),
            ),
            link,
        )
        for link in raw_links[:MAX_SCORED_LINKS]
    ]
    if sort_by_score:
        # Stable sort on the negated score keeps document order within ties,
        # so equal-scoring links stay in the order the page declared them.
        scored.sort(key=lambda pair: -pair[0])
    return [
        LinkInfo(
            text=_clean_clamp(link.get("text", ""), MAX_FIELD_LENGTH),
            url=_clean_clamp(link.get("url", ""), MAX_FIELD_LENGTH),
            rel=_clean_clamp_opt(link.get("rel"), MAX_ATTR_LENGTH),
            type=_clean_clamp_opt(link.get("type"), MAX_ATTR_LENGTH),
            score=score,
        )
        for score, link in scored[:MAX_LINKS]
    ]


def _convert_images(raw_images: list[dict]) -> list[ImageInfo]:
    """Convert raw image dictionaries to ImageInfo models.

    Args:
        raw_images: List of dictionaries with image data.

    Returns:
        List of ImageInfo Pydantic models.
    """
    return [
        ImageInfo(
            src=_clean_clamp(img.get("src", ""), MAX_FIELD_LENGTH),
            alt=_clean_clamp(img.get("alt", ""), MAX_FIELD_LENGTH),
            title=_clean_clamp(img.get("title", ""), MAX_IMAGE_TITLE_LENGTH),
            width=img.get("width"),
            height=img.get("height"),
            srcset=_clamp_opt(img.get("srcset"), MAX_FIELD_LENGTH),
            content_base64=img.get("content_base64"),
            mime_type=_clean_clamp_opt(img.get("mime_type"), MAX_ATTR_LENGTH),
            byte_size=img.get("byte_size"),
        )
        for img in raw_images[:MAX_IMAGES]
    ]


@router.post(
    "/url",
    operation_id="scrape_url",
    summary="Scrape a single URL",
    response_model=ScrapeResponse,
    responses=SCRAPE_RESPONSES,
)
async def scrape_url(
    request: ScrapeRequest, http_request: Request, response: Response
) -> ScrapeResponse:
    """Scrape a single URL and extract content.

    Uses Playwright for JavaScript-heavy sites or simple HTTP for static pages.
    Supports screenshot capture, custom headers, cookies, and viewport settings.

    Args:
        request: Scrape request containing URL and configuration options.
        token: JWT token payload for authentication.

    Returns:
        ScrapeResponse containing extracted content, metadata, links, and images.

    Raises:
        HTTPException: 500 error if scraping fails due to network or parsing errors.
    """
    logger.info(f"Received scrape request for URL: {request.url}")

    # Result cache: an anonymous, opted-in repeat scrape returns the prior
    # response without re-fetching. Checked BEFORE the SSRF guard so a HIT does no
    # network at all — safe because only an already-guarded COMPLETED scrape could
    # have been stored.
    cache = get_result_cache()
    cache_eligible = cache.cacheable(request)
    cache_key = cache.make_key(request) if cache_eligible else None
    if cache_key is not None:
        cached = cache.get(cache_key)
        if cached is not None:
            record_cache("hit")
            response.headers["X-Cache"] = "HIT"
            return cached

    await _guard_url(str(request.url))
    if request.auth_config and request.auth_config.login_url:
        await _guard_url(str(request.auth_config.login_url), field="auth_config.login_url")

    scraper_manager = get_scraper_manager()

    service_auth = await _resolve_auth_config(request.auth_config, http_request)

    try:
        config = ScrapeConfig(
            timeout=request.timeout,
            delay=request.delay,
            screenshot=request.screenshot,
            download_images=request.download_images,
            user_agent=request.user_agent,
            headers=request.headers or {},
            cookies=request.cookies or {},
            javascript_enabled=request.javascript_enabled,
            link_query=request.link_query,
            virtual_scroll=request.virtual_scroll,
            stealth=request.stealth,
            auth_config=service_auth,
        )

        result = await scraper_manager.scrape_url(
            url=str(request.url),
            scraper_type=convert_scraper_type(request.scraper_type),
            config=config,
            viewport_width=request.viewport_width,
            viewport_height=request.viewport_height,
            wait_until=request.wait_until,
            wait_for_selector=request.wait_for_selector,
            screenshot=request.screenshot,
            download_images=request.download_images,
        )

        metadata: dict[str, Any] = dict(result.metadata or {})
        extracted, extraction_truncated, extraction_field_errors = _run_extraction(
            result.html_content, request.extraction_schema
        )
        if extraction_truncated:
            # A short list that looks complete is the failure mode here: the
            # caller must be able to tell 1,000-of-5,000 from "that was all".
            metadata["extraction_truncated"] = True
        if extraction_field_errors:
            # These fields did NOT simply match nothing — their (already
            # CSS-valid) selector raised while running against this page. That
            # must not look like an empty/default field with no trace.
            metadata["extraction_field_errors"] = extraction_field_errors
        built = ScrapeResponse(
            url=_clean_clamp(result.url, MAX_FIELD_LENGTH),
            final_url=_clean_clamp_opt(metadata.get("final_url"), MAX_FIELD_LENGTH),
            html_content=_clamp_opt(result.html_content, MAX_CONTENT_LENGTH),
            text_content=_clamp_opt(result.text_content, MAX_CONTENT_LENGTH),
            markdown_content=_clamp_opt(result.markdown_content, MAX_CONTENT_LENGTH),
            title=_clean_clamp_opt(result.title, MAX_FIELD_LENGTH),
            metadata=_sanitize_metadata(metadata),
            structured_data=result.structured_data or {},
            extracted_data=extracted,
            links=_convert_links(result.links, request.link_query, request.sort_links_by_score),
            images=_convert_images(result.images),
            screenshot_base64=_clamp_opt(result.screenshot_base64, MAX_SCREENSHOT_LENGTH),
            status=convert_scrape_status(result.status),
            error=_clean_clamp_opt(result.error, MAX_FIELD_LENGTH),
            scraper_used=_clean_clamp(result.scraper_used or "unknown", MAX_ATTR_LENGTH),
            processing_time_seconds=result.processing_time_seconds,
            storage_state=result.storage_state,
        )

        # Store only COMPLETED responses; mark HIT/MISS/BYPASS for observability.
        if cache_key is not None:
            if result.status == ServiceScrapeStatus.COMPLETED:
                cache.set(cache_key, built)
            record_cache("miss")
            response.headers["X-Cache"] = "MISS"
        else:
            record_cache("bypass")
            response.headers["X-Cache"] = "BYPASS"

        return built

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error scraping URL {request.url}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Scraping failed: {str(e)}",
        )


@router.post(
    "/bulk",
    operation_id="scrape_bulk",
    summary="Scrape multiple URLs in one request",
    response_model=BulkScrapeResponse,
    responses=SCRAPE_RESPONSES,
)
async def scrape_bulk(request: BulkScrapeRequest, http_request: Request) -> BulkScrapeResponse:
    """Scrape multiple URLs in a single request.

    Processes URLs sequentially to avoid overwhelming target servers.
    Each URL is scraped independently and failures don't affect other URLs.

    Args:
        request: Bulk scrape request containing list of URLs and options.
        token: JWT token payload for authentication.

    Returns:
        BulkScrapeResponse with individual results for each URL and summary stats.

    Raises:
        HTTPException: 500 error if bulk scraping fails catastrophically.
    """
    logger.info(f"Received bulk scrape request for {len(request.urls)} URLs")

    # A bad login_url poisons the whole batch (shared auth) → reject up front;
    # individual blocked target URLs become per-URL FAILED results in the manager.
    if request.auth_config and request.auth_config.login_url:
        await _guard_url(str(request.auth_config.login_url), field="auth_config.login_url")

    start_time = time.time()
    scraper_manager = get_scraper_manager()

    service_auth = await _resolve_auth_config(request.auth_config, http_request)

    try:
        config = ScrapeConfig(
            timeout=request.timeout,
            delay=request.delay,
            screenshot=request.screenshot,
            download_images=request.download_images,
            follow_links=request.follow_links,
            max_depth=request.max_depth,
            javascript_enabled=request.javascript_enabled,
            link_query=request.link_query,
            allowed_domains=request.allowed_domains,
            virtual_scroll=request.virtual_scroll,
            stealth=request.stealth,
            auth_config=service_auth,
        )

        urls = [str(url) for url in request.urls]

        results = await scraper_manager.scrape_bulk(
            urls=urls,
            scraper_type=convert_scraper_type(request.scraper_type),
            config=config,
            screenshot=request.screenshot,
            download_images=request.download_images,
            wait_until=request.wait_until,
            wait_for_selector=request.wait_for_selector,
        )

        response_results: list[ScrapeResponse] = []
        completed = 0
        failed = 0

        for result in results:
            bulk_extracted, bulk_truncated, bulk_field_errors = _run_extraction(
                result.html_content, request.extraction_schema
            )
            bulk_metadata: dict[str, Any] = dict(_sanitize_metadata(result.metadata or {}))
            if bulk_truncated:
                bulk_metadata["extraction_truncated"] = True
            if bulk_field_errors:
                bulk_metadata["extraction_field_errors"] = bulk_field_errors
            response_results.append(
                ScrapeResponse(
                    url=_clean_clamp(result.url, MAX_FIELD_LENGTH),
                    final_url=_clean_clamp_opt(
                        result.metadata.get("final_url") if result.metadata else None,
                        MAX_FIELD_LENGTH,
                    ),
                    html_content=_clamp_opt(result.html_content, MAX_CONTENT_LENGTH),
                    text_content=_clamp_opt(result.text_content, MAX_CONTENT_LENGTH),
                    markdown_content=_clamp_opt(result.markdown_content, MAX_CONTENT_LENGTH),
                    title=_clean_clamp_opt(result.title, MAX_FIELD_LENGTH),
                    metadata=bulk_metadata,
                    structured_data=result.structured_data or {},
                    extracted_data=bulk_extracted,
                    links=_convert_links(
                        result.links, request.link_query, request.sort_links_by_score
                    ),
                    images=_convert_images(result.images),
                    screenshot_base64=_clamp_opt(result.screenshot_base64, MAX_SCREENSHOT_LENGTH),
                    status=convert_scrape_status(result.status),
                    error=_clean_clamp_opt(result.error, MAX_FIELD_LENGTH),
                    scraper_used=_clean_clamp(result.scraper_used or "unknown", MAX_ATTR_LENGTH),
                    processing_time_seconds=result.processing_time_seconds,
                    storage_state=result.storage_state,
                )
            )

            if result.status == ServiceScrapeStatus.COMPLETED:
                completed += 1
            else:
                failed += 1

        total_time = time.time() - start_time

        return BulkScrapeResponse(
            total_urls=len(request.urls),
            completed=completed,
            failed=failed,
            results=response_results,
            processing_time_seconds=total_time,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in bulk scrape: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Bulk scraping failed: {str(e)}",
        )
