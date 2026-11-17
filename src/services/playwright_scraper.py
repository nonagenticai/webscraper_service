"""Playwright-based web scraper for JavaScript-heavy sites.

Uses Playwright to render JavaScript and extract content from dynamic pages.
Supports screenshot capture, custom headers, cookies, and full page rendering.
"""

import asyncio
import base64
import logging
import time
from typing import Any, Literal, cast

from playwright.async_api import (
    Browser,
    BrowserContext,
    Page,
    Playwright,
    ProxySettings,
    async_playwright,
)
from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from playwright_stealth import Stealth

from config import get_settings
from services.base_scraper import AuthConfig, BaseScraper, ScrapeConfig, ScrapeResult, ScrapeStatus
from services.content_extraction import extract_main_text
from services.image_utils import enrich_images_with_content
from services.url_guard import UrlNotAllowedError, assert_allowed_url
from services.virtual_scroll import apply_virtual_scroll

logger = logging.getLogger(__name__)
settings = get_settings()

DEFAULT_USERNAME_SELECTORS = (
    'input[name="email"], input[name="username"], '
    'input[type="email"], input[id*="email"], input[id*="user"]'
)
DEFAULT_PASSWORD_SELECTORS = 'input[name="password"], input[type="password"]'
DEFAULT_SUBMIT_SELECTORS = (
    'button[type="submit"], button:has-text("Sign in"), '
    'button:has-text("Log in"), input[type="submit"]'
)

_DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# Types that need an explicit BrowserContext (storage_state injection or
# http_credentials). Bearer auth only sets headers on the default page.
_CONTEXT_AUTH_TYPES = {"form", "storage_state", "basic"}

# Headroom on top of the navigation timeout for the post-load extraction phase
# (content/title/evaluate/screenshot/image-download). The overall wall-clock
# budget = effective_timeout + this, and is the backstop for page.evaluate (which
# has no timeout parameter and is not governed by page.set_default_timeout).
EXTRACTION_MARGIN_SECONDS = 30.0

# Slice of the remaining budget kept back from virtual scrolling for the work
# that MUST still happen after it (content, text, metadata, links, images,
# screenshot). Scrolling until the deadline would return a fully-materialized
# page that then dies in extraction — the worst of both.
_SCROLL_RESERVE_SECONDS = 10.0

# Hard cap on a best-effort page/context close so a wedged browser object cannot
# re-hang the request during cleanup.
_CLOSE_TIMEOUT_SECONDS = 5.0


class PlaywrightScraper(BaseScraper):
    """Playwright-based scraper for JavaScript-heavy websites.

    Provides full JavaScript rendering, screenshot capture, and
    comprehensive content extraction from dynamic web pages.

    Attributes:
        headless: Whether to run browser in headless mode.
        playwright: Playwright instance.
        browser: Browser instance.
        _initialized: Whether browser has been initialized.
    """

    def __init__(self, config: ScrapeConfig | None = None, headless: bool = True) -> None:
        """Initialize the Playwright scraper.

        Args:
            config: Optional scraping configuration with timeout, headers, etc.
            headless: Whether to run browser without GUI (default True).
        """
        super().__init__(config)
        self.headless = headless
        self.playwright: Playwright | None = None
        self.browser: Browser | None = None
        self._initialized = False
        # Serializes browser launch: scrape() (and thus _ensure_browser) runs
        # OUTSIDE the manager's _scraper_lock, so two concurrent first-requests
        # would otherwise both pass the `not _initialized` check and each
        # launch() a Chromium, orphaning the first browser + node driver.
        self._init_lock = asyncio.Lock()

    def is_initialized(self) -> bool:
        """Check if the browser has been initialized.

        Returns:
            True if browser is initialized, False otherwise.
        """
        return self._initialized

    async def _discard_dead_browser(self) -> None:
        """Best-effort teardown of a browser that is already gone.

        Every step is individually guarded: the whole point is that this object
        is in an unknown state, and a raise here would leave ``_initialized``
        True and block the relaunch it exists to enable.
        """
        for name, closeable in (("browser", self.browser), ("playwright", self.playwright)):
            if closeable is None:
                continue
            try:
                stop = getattr(closeable, "close", None) or getattr(closeable, "stop", None)
                if stop is not None:
                    await asyncio.wait_for(stop(), timeout=_CLOSE_TIMEOUT_SECONDS)
            except Exception as e:
                logger.warning(f"Discarding dead {name} raised (ignored): {e}")
        self.browser = None
        self.playwright = None
        self._initialized = False

    def is_healthy(self) -> bool:
        """Initialized AND the browser process is still connected.

        ``_initialized`` only records that a launch once succeeded; a Chromium
        that has since died (OOM-killed under memory pressure is the realistic
        case on this pod) leaves it True. Readiness has to reflect the process
        that is actually there, not the one that started.
        """
        if not self._initialized:
            return False
        browser = self.browser
        if browser is None:
            return False
        try:
            return bool(browser.is_connected())
        except Exception as e:  # a driver in the middle of dying
            logger.warning(f"Browser connectivity check failed: {e}")
            return False

    async def _ensure_browser(self) -> None:
        """Ensure browser is initialized and ready for scraping.

        Lazily initializes Playwright and launches a Chromium browser with
        optimized settings for headless scraping. Guarded by ``_init_lock`` with
        double-checked ``is_healthy()`` so concurrent cold-start requests launch
        exactly one browser. A launch failure tears down any partially-started
        Playwright so a later request can cleanly retry.

        Gating on ``is_healthy()`` rather than ``_initialized`` is what makes a
        crashed Chromium RECOVERABLE. ``_initialized`` stays True for a browser
        that has died, so every later scrape used a dead handle and — now that
        readiness reflects the browser — the pod would have parked itself
        NotReady forever with nothing able to relaunch.
        """
        if self.is_healthy():
            return
        async with self._init_lock:
            if self.is_healthy():  # another request won the race
                return
            if self._initialized:
                # A previous browser died. Tear the corpse down before
                # relaunching, or the old node driver leaks for the pod's life.
                logger.warning("Playwright browser is no longer connected; relaunching")
                await self._discard_dead_browser()
            logger.info("Initializing Playwright browser...")
            try:
                self.playwright = await async_playwright().start()
                # Optional ops-level egress proxy (browser-wide). None → direct.
                proxy: ProxySettings | None = (
                    {"server": settings.outbound_proxy} if settings.outbound_proxy else None
                )
                self.browser = await self.playwright.chromium.launch(
                    headless=self.headless,
                    proxy=proxy,
                    args=[
                        "--no-sandbox",
                        "--disable-setuid-sandbox",
                        "--disable-dev-shm-usage",
                        "--disable-accelerated-2d-canvas",
                        "--disable-gpu",
                    ],
                )
            except Exception:
                # Don't leak a half-started Playwright (node driver) on failure.
                if self.playwright is not None:
                    try:
                        await self.playwright.stop()
                    except Exception as stop_err:
                        logger.warning(f"Failed to stop partial Playwright: {stop_err}")
                self.playwright = None
                self.browser = None
                self._initialized = False
                raise
            self._initialized = True
            logger.info("Playwright browser initialized")

    async def scrape(self, url: str, **kwargs: Any) -> ScrapeResult:
        """Scrape a URL using Playwright with full JavaScript rendering.

        Args:
            url: The URL to scrape.
            **kwargs: Additional options:
                - viewport_width: Browser viewport width (default 1920).
                - viewport_height: Browser viewport height (default 1080).
                - wait_until: Page load condition ('load', 'domcontentloaded',
                  'networkidle').
                - screenshot: Whether to capture a screenshot.
                - wait_for_selector: CSS selector to wait for before extraction.

        Returns:
            ScrapeResult with extracted HTML, text, links, images, and metadata.
        """
        start_time = time.time()

        # Request-local config: read every per-request field (auth/cookies/headers/
        # user_agent/delay/screenshot/...) from cfg, NEVER self.config, which a
        # concurrent request may have mutated on the shared instance.
        cfg = self._resolve_config(kwargs)

        try:
            await self._ensure_browser()
            if self.browser is None:
                raise RuntimeError("Browser not initialized after _ensure_browser()")

            viewport_width: int = kwargs.get("viewport_width", 1920)
            viewport_height: int = kwargs.get("viewport_height", 1080)
            is_fallback: bool = kwargs.get("is_fallback", False)
            # Default to "load" (not "networkidle"): on live SPAs with analytics /
            # websockets / polling, networkidle's 500ms-quiet window never settles,
            # so goto blocks the whole timeout on a perfectly good page. networkidle
            # stays available as an explicit opt-in via the wait_until kwarg.
            default_wait = "domcontentloaded" if is_fallback else "load"
            wait_until = cast(
                Literal["commit", "domcontentloaded", "load", "networkidle"] | None,
                kwargs.get("wait_until", default_wait) if not is_fallback else default_wait,
            )
            capture_screenshot: bool = kwargs.get("screenshot", cfg.screenshot)
            wait_for_selector: str | None = kwargs.get("wait_for_selector")

            # Effective timeout is request-local: prefer the explicit per-call value
            # (passed by ScraperManager) over cfg.timeout. Governs both the
            # navigation timeout and the overall wall-clock budget.
            effective_timeout: float = float(kwargs.get("timeout") or cfg.timeout)
            nav_timeout_ms = int(effective_timeout * 1000)
            extraction_margin: float = float(
                kwargs.get("extraction_margin", EXTRACTION_MARGIN_SECONDS)
            )
            overall_budget = effective_timeout + extraction_margin

            logger.info(f"Scraping URL with Playwright: {url}")

            auth = cfg.auth_config
            user_agent = cfg.user_agent or _DEFAULT_USER_AGENT
            viewport = {"width": viewport_width, "height": viewport_height}

            # Create page (+ auth context). _open_page cleans up the context if
            # page creation fails, so a creation error never leaks a context (the
            # inner try/finally below only runs once page+context are bound).
            page, context = await self._open_page(auth, viewport, user_agent, cfg.stealth)

            # Page/context are now bound — any failure setting up routing or
            # timeouts below (before the inner try/finally takes over) must still
            # close them, or the browser context leaks. Close-on-error here.
            try:
                # SSRF: the browser follows redirects natively (incl. on the
                # login_url path), so the manager's initial-URL guard can't see a
                # 302 → internal/metadata host. Guard EVERY request the page
                # issues — not just top-level navigations. A scraped (attacker-
                # influenced) page can `fetch()`/XHR/load a subresource from an
                # internal ClusterIP or the cloud metadata endpoint
                # (169.254.169.254); those are blind-SSRF (and read-exfil where
                # CORS permits) vectors that never surface as a navigation
                # request. Abort any request whose target host is disallowed,
                # regardless of request type.
                if settings.ssrf_protection_enabled:
                    allowed_hosts = tuple(settings.ssrf_allowed_hosts)

                    async def _ssrf_route(route: Any) -> None:
                        req = route.request
                        try:
                            await asyncio.to_thread(assert_allowed_url, req.url, allowed_hosts)
                        except UrlNotAllowedError:
                            await route.abort("blockedbyclient")
                            return
                        await route.continue_()

                    await page.route("**/*", _ssrf_route)

                # Bound EVERY page operation. content()/title()/wait_for_*
                # otherwise rely on Playwright's implicit default; this makes the
                # bound explicit and request-local. (page.evaluate ignores this —
                # the overall budget below is its backstop.)
                page.set_default_timeout(nav_timeout_ms)
                page.set_default_navigation_timeout(nav_timeout_ms)
            except Exception:
                await self._safe_close(page)
                if context is not None:
                    await self._safe_close(context)
                raise

            async def _do_scrape() -> ScrapeResult:
                # Merge Authorization header for bearer auth on top of any custom headers.
                extra_headers: dict[str, str] = dict(cfg.headers or {})
                if auth is not None and auth.type == "bearer" and auth.token:
                    extra_headers["Authorization"] = f"Bearer {auth.token}"
                if extra_headers:
                    await page.set_extra_http_headers(extra_headers)
                    logger.debug(f"Extra headers configured: {list(extra_headers.keys())}")

                if cfg.cookies:
                    if not isinstance(cfg.cookies, dict):
                        logger.warning(f"Invalid cookies type: {type(cfg.cookies)}, expected dict")
                    else:
                        cookies: list[dict[str, Any]] = []
                        for name, value in cfg.cookies.items():
                            if name and isinstance(name, str) and value is not None:
                                cookies.append({"name": name, "value": str(value), "url": url})
                        if cookies:
                            await page.context.add_cookies(cast(Any, cookies))
                            logger.debug(f"Cookies configured: {len(cookies)} cookies")

                # If form auth has an explicit login_url, log in BEFORE navigating
                # to the target. Otherwise the site's own redirect-to-login on the
                # target URL would land us on the wrong page and we'd fill the form
                # against a redirected-to login form (or capture the login page as
                # the result) instead of authenticating cleanly first.
                response = None
                if auth is not None and auth.type == "form" and auth.login_url:
                    await self._perform_form_auth(
                        page,
                        auth,
                        original_url=url,
                        renavigate_after=False,
                        nav_timeout_ms=nav_timeout_ms,
                    )
                    response = await page.goto(url, wait_until=wait_until, timeout=nav_timeout_ms)
                else:
                    response = await page.goto(url, wait_until=wait_until, timeout=nav_timeout_ms)
                    if auth is not None and auth.type == "form":
                        await self._perform_form_auth(
                            page, auth, original_url=url, nav_timeout_ms=nav_timeout_ms
                        )

                if wait_for_selector:
                    await page.wait_for_selector(wait_for_selector, timeout=nav_timeout_ms)

                if cfg.delay > 0:
                    await page.wait_for_timeout(int(cfg.delay * 1000))

                # Materialize lazily-loaded content BEFORE reading the DOM, so
                # every downstream consumer (text, markdown, links, images,
                # extraction_schema) sees the full page rather than the first
                # viewport. Stats are surfaced in metadata: a caller needs to
                # tell "the feed ended" apart from "we hit the step cap and the
                # page probably has more".
                scroll_stats: dict[str, Any] | None = None
                if cfg.virtual_scroll is not None:
                    # Scrolling shares the scrape's single deadline. 30 steps x
                    # 5s of wait is a VALID config that on its own outruns the
                    # default 30s budget — and overrunning raises, discarding a
                    # perfectly good page. Hand the scroll only the time that is
                    # actually left, minus the extraction reserve.
                    scroll_budget = (
                        overall_budget - (time.time() - start_time) - _SCROLL_RESERVE_SECONDS
                    )
                    scroll_stats = await apply_virtual_scroll(
                        page, cfg.virtual_scroll, budget_seconds=max(0.0, scroll_budget)
                    )

                html_content = await page.content()
                # Absent title → None (page.title() returns "") to match httpx/scrapy.
                title = (await page.title()) or None

                # Extract clean main content from the rendered DOM HTML so the
                # Playwright and httpx scrapers emit identical text_content.
                text_content = extract_main_text(html_content, url)

                metadata = await self._extract_metadata(page)
                if scroll_stats is not None:
                    metadata["virtual_scroll"] = scroll_stats
                # Authoritative transport values LAST so scraped <meta> tags
                # can't clobber them.
                metadata["status_code"] = response.status if response else None
                metadata["final_url"] = page.url

                links = await self._extract_links(page, url)
                images = await self._extract_images(page)

                # Download image content if requested
                download_images: bool = kwargs.get("download_images", cfg.download_images)
                if download_images and images:
                    images = await enrich_images_with_content(images)

                screenshot_base64: str | None = None
                if capture_screenshot:
                    screenshot_base64 = await self._capture_screenshot_base64(page)

                storage_state: dict[str, Any] | None = None
                if auth is not None:
                    try:
                        storage_state = cast(dict[str, Any], await page.context.storage_state())
                    except Exception as e:
                        logger.warning(f"Failed to capture storage_state: {e}")

                processing_time = time.time() - start_time

                logger.info(f"Successfully scraped {url} in {processing_time:.2f}s")

                return ScrapeResult(
                    url=url,
                    html_content=html_content,
                    text_content=text_content,
                    title=title,
                    metadata=metadata,
                    links=links,
                    images=images,
                    screenshot_base64=screenshot_base64,
                    status=ScrapeStatus.COMPLETED,
                    error=None,
                    scraper_used=self.name,
                    processing_time_seconds=processing_time,
                    storage_state=storage_state,
                )

            try:
                return await asyncio.wait_for(_do_scrape(), timeout=overall_budget)

            except PlaywrightTimeoutError as e:
                logger.error(f"Timeout scraping {url}: {e}")
                return ScrapeResult(
                    url=url,
                    status=ScrapeStatus.FAILED,
                    error=f"Timeout: {str(e)}",
                    scraper_used=self.name,
                    processing_time_seconds=time.time() - start_time,
                )

            except TimeoutError:
                # asyncio.wait_for raises TimeoutError when the overall budget is
                # exceeded — the backstop for an unbounded page.evaluate that
                # page.set_default_timeout cannot govern.
                logger.error(f"Overall scrape budget ({overall_budget:.1f}s) exceeded for {url}")
                return ScrapeResult(
                    url=url,
                    status=ScrapeStatus.FAILED,
                    error=f"overall scrape budget exceeded after {overall_budget:.1f}s",
                    scraper_used=self.name,
                    processing_time_seconds=time.time() - start_time,
                )

            finally:
                await self._safe_close(page)
                if context is not None:
                    await self._safe_close(context)

        except Exception as e:
            logger.error(f"Error scraping {url} with Playwright: {e}", exc_info=True)
            return ScrapeResult(
                url=url,
                status=ScrapeStatus.FAILED,
                error=str(e),
                scraper_used=self.name,
                processing_time_seconds=time.time() - start_time,
            )

    async def _open_page(
        self,
        auth: AuthConfig | None,
        viewport: dict[str, int],
        user_agent: str,
        stealth: bool = False,
    ) -> tuple[Page, BrowserContext | None]:
        """Open a Page, creating an auth BrowserContext when required.

        Returns ``(page, context)`` where ``context`` is None for the
        no-context path. If ``new_page()`` fails after the context is created,
        the context is closed before re-raising so it is never orphaned (the
        caller's try/finally only covers the post-creation lifecycle).
        """
        if self.browser is None:  # pragma: no cover — caller ensures browser
            raise RuntimeError("Browser not initialized")

        if auth is not None and auth.type in _CONTEXT_AUTH_TYPES:
            context = await self._create_auth_context(auth, viewport, user_agent)
            try:
                page = await context.new_page()
            except Exception:
                await self._safe_close(context)
                raise
            await self._maybe_apply_stealth(page, stealth)
            return page, context

        page = await self.browser.new_page(viewport=viewport, user_agent=user_agent)
        await self._maybe_apply_stealth(page, stealth)
        return page, None

    async def _maybe_apply_stealth(self, page: Page, stealth: bool) -> None:
        """Patch the page's automation fingerprint, best-effort.

        Applied BEFORE navigation — the patches are init scripts, so applying
        them afterwards would leave the first document unpatched, which is the
        only document that matters. A failure here degrades to an unpatched
        page rather than failing the scrape: stealth is a reach improvement,
        never a correctness requirement.
        """
        if not stealth:
            return
        try:
            await Stealth().apply_stealth_async(page)
            logger.debug("stealth patches applied to page")
        except Exception as e:
            logger.warning(f"stealth patches failed, continuing unpatched: {e}")

    async def _safe_close(self, closeable: Page | BrowserContext) -> None:
        """Close a page/context best-effort within a hard time bound.

        A wedged browser object's ``close()`` can itself block; bounding it keeps
        cleanup (which runs in the scrape ``finally``, including during an
        overall-budget cancellation) from re-hanging the request. A page that
        won't close is logged and abandoned — the next request lazily re-inits
        the browser via ``_ensure_browser``.
        """
        try:
            await asyncio.wait_for(closeable.close(), timeout=_CLOSE_TIMEOUT_SECONDS)
        except Exception as e:
            logger.warning(f"Failed to close {type(closeable).__name__} cleanly: {e}")

    async def _create_auth_context(
        self,
        auth: AuthConfig,
        viewport: dict[str, int],
        user_agent: str,
    ) -> BrowserContext:
        """Create a BrowserContext pre-configured for the auth type.

        ``storage_state`` → inject the cached Playwright storageState.
        ``basic`` → set http_credentials so Playwright handles Basic auth automatically.
        ``form`` → plain context; login is performed by ``_perform_form_auth``.
        """
        if self.browser is None:
            raise RuntimeError("Browser not initialized")

        ctx_kwargs: dict[str, Any] = {"viewport": viewport, "user_agent": user_agent}
        if auth.type == "storage_state" and auth.storage_state:
            ctx_kwargs["storage_state"] = auth.storage_state
        elif auth.type == "basic":
            ctx_kwargs["http_credentials"] = {
                "username": auth.username,
                "password": auth.password,
            }
        return await self.browser.new_context(**ctx_kwargs)

    async def _perform_form_auth(
        self,
        page: Page,
        auth: AuthConfig,
        original_url: str,
        renavigate_after: bool = True,
        nav_timeout_ms: int | None = None,
    ) -> None:
        """Fill and submit a login form, then wait for the post-login state.

        If the app redirected us away from ``original_url`` during/after login,
        re-navigate to ``original_url`` so content extraction pulls the page
        the caller actually asked for — not whatever dashboard the app landed on.

        ``nav_timeout_ms`` is the request-local navigation budget for the
        post-login re-navigation; it falls back to the instance config only when
        not supplied (legacy callers).

        Credentials are never logged.
        """
        renav_timeout_ms = nav_timeout_ms or int(self.config.timeout * 1000)
        if auth.login_url and auth.login_url != page.url:
            logger.debug(f"Navigating to explicit login_url: {auth.login_url}")
            # The login page LOAD uses the request-local navigation budget (same as
            # the post-login renav below); auth.wait_timeout_ms governs the
            # post-submit *condition* waits, not page loads.
            await page.goto(auth.login_url, wait_until="networkidle", timeout=renav_timeout_ms)

        username_sel = auth.username_selector or DEFAULT_USERNAME_SELECTORS
        password_sel = auth.password_selector or DEFAULT_PASSWORD_SELECTORS
        submit_sel = auth.submit_selector or DEFAULT_SUBMIT_SELECTORS

        await page.locator(username_sel).first.fill(auth.username)
        await page.locator(password_sel).first.fill(auth.password)
        await page.locator(submit_sel).first.click()

        if auth.wait_for_url:
            await page.wait_for_url(auth.wait_for_url, timeout=auth.wait_timeout_ms)
        elif auth.wait_for_selector:
            await page.wait_for_selector(auth.wait_for_selector, timeout=auth.wait_timeout_ms)
        else:
            # Default wait: poll in JS, not via page.wait_for_url(callable).
            # Playwright's callable predicate for wait_for_url only fires on
            # navigation events, so SPA login forms that set a cookie via
            # fetch() (same-document, no navigation) would time out. Polling
            # window.location inside the page observes both full navigations
            # and same-document (history.pushState / replaceState) changes.
            await page.wait_for_function(
                "() => !window.location.pathname.toLowerCase().includes('login')",
                timeout=auth.wait_timeout_ms,
            )

        if renavigate_after and page.url != original_url:
            logger.debug(f"Post-login URL {page.url} != target {original_url}; re-navigating")
            await page.goto(
                original_url,
                wait_until="networkidle",
                timeout=renav_timeout_ms,
            )

    async def _extract_metadata(self, page: Page) -> dict[str, Any]:
        """Extract metadata from page meta tags and link elements.

        Args:
            page: Playwright Page instance.

        Returns:
            Dictionary of metadata including Open Graph, description, etc.
        """
        try:
            metadata: dict[str, Any] = await page.evaluate(
                """() => {
                const meta = {};
                document.querySelectorAll('meta').forEach(tag => {
                    const name = tag.getAttribute('name') || tag.getAttribute('property');
                    const content = tag.getAttribute('content');
                    if (name && content) {
                        meta[name] = content;
                    }
                });
                const canonical = document.querySelector('link[rel="canonical"]');
                if (canonical) {
                    meta['canonical_url'] = canonical.href;
                }
                return meta;
            }"""
            )
            return metadata
        except Exception as e:
            logger.warning(f"Error extracting metadata: {e}")
            return {}

    async def _extract_links(self, page: Page, base_url: str) -> list[dict[str, Any]]:
        """Extract all anchor links from the page.

        Args:
            page: Playwright Page instance.
            base_url: Base URL for classifying internal vs external links.

        Returns:
            List of link dictionaries with text, url, rel, and type.
        """
        try:
            links: list[dict[str, Any]] = await page.evaluate(
                """() => {
                const links = [];
                document.querySelectorAll('a[href]').forEach(a => {
                    links.push({
                        text: a.innerText.trim(),
                        url: a.href,
                        rel: a.rel || null,
                    });
                });
                return links;
            }"""
            )

            base_domain = self.extract_domain(base_url)
            for link in links:
                link_domain = self.extract_domain(link["url"])
                link["type"] = "internal" if link_domain == base_domain else "external"

            return links
        except Exception as e:
            logger.warning(f"Error extracting links: {e}")
            return []

    async def _extract_images(self, page: Page) -> list[dict[str, Any]]:
        """Extract all images from the page, including lazy-loaded sources.

        Args:
            page: Playwright Page instance.

        Returns:
            List of image dictionaries with src, alt, title, width, height, srcset.
        """
        try:
            images: list[dict[str, Any]] = await page.evaluate(
                """() => {
                const images = [];
                document.querySelectorAll('img').forEach(img => {
                    const src = img.src
                        || img.dataset.src
                        || img.dataset.original
                        || img.dataset.lazySrc
                        || '';
                    images.push({
                        src: src,
                        alt: img.alt || '',
                        title: img.title || '',
                        width: img.naturalWidth || null,
                        height: img.naturalHeight || null,
                        srcset: img.srcset || null,
                    });
                });
                // Also extract <picture><source> elements
                document.querySelectorAll('picture').forEach(picture => {
                    const img = picture.querySelector('img');
                    if (!img) return;
                    const sources = picture.querySelectorAll('source');
                    let bestSrcset = '';
                    sources.forEach(s => {
                        if (s.srcset) bestSrcset = s.srcset;
                    });
                    const picSrc = img.src
                        || img.dataset.src
                        || img.dataset.original
                        || img.dataset.lazySrc
                        || '';
                    const existing = images.find(i => i.src === picSrc);
                    if (existing) {
                        // Update srcset from <source> if missing
                        if (bestSrcset && !existing.srcset) {
                            existing.srcset = bestSrcset;
                        }
                    } else {
                        images.push({
                            src: picSrc,
                            alt: img.alt || '',
                            title: img.title || '',
                            width: img.naturalWidth || null,
                            height: img.naturalHeight || null,
                            srcset: bestSrcset || img.srcset || null,
                        });
                    }
                });
                return images;
            }"""
            )
            return images
        except Exception as e:
            logger.warning(f"Error extracting images: {e}")
            return []

    async def _capture_screenshot_base64(self, page: Page) -> str | None:
        """Capture full page screenshot and encode as base64.

        Args:
            page: Playwright Page instance.

        Returns:
            Base64-encoded screenshot string, or None if capture fails.
        """
        try:
            screenshot_bytes = await page.screenshot(full_page=True)
            screenshot_base64 = base64.b64encode(screenshot_bytes).decode("utf-8")
            logger.info("Screenshot captured successfully")
            return screenshot_base64
        except Exception as e:
            logger.error(f"Error capturing screenshot: {e}")
            return None

    async def close(self) -> None:
        """Close browser and release all Playwright resources.

        Should be called during shutdown to ensure proper cleanup.
        """
        logger.info("Closing Playwright browser...")

        if self.browser:
            await self.browser.close()
            self.browser = None

        if self.playwright:
            await self.playwright.stop()
            self.playwright = None

        self._initialized = False
        logger.info("Playwright browser closed")


__all__ = ["PlaywrightScraper"]
