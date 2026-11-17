"""Scraper manager for orchestrating web scraping operations.

Provides high-level interface for scraping with automatic
scraper selection, configuration management, and error handling.
"""

import asyncio
import copy
import logging
from dataclasses import replace
from typing import Any

import httpx

from config import get_settings
from services.base_scraper import (
    AuthConfig,
    BaseScraper,
    ScrapeConfig,
    ScrapeResult,
    ScraperType,
    ScrapeStatus,
)
from services.content_extraction import extract_markdown
from services.curl_cffi_scraper import CurlCffiScraper
from services.httpx_scraper import HttpxScraper
from services.metrics import record_scrape
from services.playwright_scraper import PlaywrightScraper
from services.scrapy_scraper import ScrapyScraper
from services.structured_data import extract_structured_data
from services.url_guard import assert_allowed_url, make_outbound_guard_hook

logger = logging.getLogger(__name__)
settings = get_settings()

_HEAD_TIMEOUT = 5.0


async def _enrich_extraction(result: ScrapeResult) -> ScrapeResult:
    """Derive LLM/RAG-ready outputs (Markdown + structured data) once, centrally.

    All backends flow through ``scrape_url``, so enriching here gives every engine
    identical ``markdown_content``/``structured_data`` without threading the work
    through each scraper. Only runs on a COMPLETED result that carries HTML; the
    (CPU-bound) parsing is offloaded so it never blocks the event loop. Best-effort
    — a derivation failure leaves the field empty, never fails the scrape.
    """
    if result.status != ScrapeStatus.COMPLETED or not result.html_content:
        return result

    base_url = (result.metadata or {}).get("final_url") or result.url
    if result.markdown_content is None:
        result.markdown_content = await asyncio.to_thread(
            extract_markdown, result.html_content, base_url
        )
    if not result.structured_data:
        result.structured_data = await asyncio.to_thread(
            extract_structured_data, result.html_content, base_url
        )
    return result


class ScraperManager:
    """High-level manager for web scraping operations.

    Manages scraper lifecycle, provides automatic scraper selection,
    and handles configuration and error management.

    Attributes:
        _playwright_scraper: Cached Playwright scraper instance.
        _httpx_scraper: Cached httpx scraper instance.
        _scrapy_scraper: Cached Scrapy scraper instance.
        _curl_cffi_scraper: Cached curl_cffi (Chrome-impersonation) scraper instance.
        _scraper_lock: Async lock for thread-safe scraper access.
    """

    def __init__(self) -> None:
        self._playwright_scraper: PlaywrightScraper | None = None
        self._httpx_scraper: HttpxScraper | None = None
        self._scrapy_scraper: ScrapyScraper | None = None
        self._curl_cffi_scraper: CurlCffiScraper | None = None
        self._scraper_lock = asyncio.Lock()
        # Global backpressure: bound concurrent scrapes so a burst can't open
        # unbounded Playwright pages/contexts (FD/memory exhaustion) on the
        # single shared browser. Makes ``scraping_max_concurrent`` (previously
        # dead config) a real limit. All scrape paths are timeout-bounded, so a
        # permit is always released.
        self._concurrency = asyncio.Semaphore(max(1, settings.scraping_max_concurrent))
        logger.info(
            f"ScraperManager initialized (max_concurrent={settings.scraping_max_concurrent})"
        )

    def is_browser_ready(self) -> bool:
        """Check if the browser is usable for scraping.

        A browser that has never been launched is not a fault — it is created
        lazily on first use — so "no scraper yet" is ready. Once one exists, it
        must still be connected: a dead Chromium means every browser-backed
        scrape will fail, and the pod should stop taking traffic rather than
        answer with first-viewport results.
        """
        if self._playwright_scraper is None:
            return True
        return bool(self._playwright_scraper.is_healthy())

    async def _head_check(self, url: str) -> int | None:
        """Quick HEAD request to pre-check URL accessibility.

        Returns:
            HTTP status code, or None if request fails/times out.
        """
        # SSRF: the HEAD pre-check also follows redirects, so guard every hop.
        event_hooks = (
            {"request": [make_outbound_guard_hook(settings.ssrf_allowed_hosts)]}
            if settings.ssrf_protection_enabled
            else {}
        )
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(_HEAD_TIMEOUT),
                verify=True,
                event_hooks=event_hooks,
                proxy=settings.outbound_proxy or None,
            ) as client:
                resp = await client.head(url, follow_redirects=True)
                return resp.status_code
        except Exception:
            return None

    async def _select_auto_scraper(
        self,
        url: str,
        config: ScrapeConfig,
        *,
        wait_for_selector: str | None = None,
        wait_until: str | None = None,
    ) -> BaseScraper:
        """Select the best scraper for AUTO mode.

        The HEAD pre-check runs WITHOUT holding ``_scraper_lock`` so concurrent
        AUTO scrapes don't serialize behind one ~5s network round-trip; the lock
        is held only for the get-or-create of the shared scraper instance.

        ``wait_for_selector`` / ``wait_until`` are per-request kwargs (not
        ScrapeConfig fields), so they are threaded in explicitly: they are
        browser-only signals and must force Playwright in AUTO mode.

        Returns the chosen scraper. (AUTO fallback to Playwright is decided by
        ``scrape_url`` from the result + the scraper's concrete type, so no
        "already tried httpx" flag is threaded back here.)
        """
        # 1. Crawling use case → Scrapy
        if config.follow_links or config.max_depth > 1:
            async with self._scraper_lock:
                return self._get_or_create_scrapy(config)

        # 2. Browser-only capabilities → Playwright. javascript_enabled,
        # virtual_scroll and stealth are browser-only; so is an explicit
        # wait_for_selector (Playwright waits for the selector to appear after
        # hydration) and any non-default wait_until (httpx does a single GET and
        # cannot honour 'networkidle'/'load'-after-JS semantics). Routing any of
        # these to httpx would silently ignore the wait and return pre-hydration
        # / first-viewport content reported as a full success.
        if (
            config.javascript_enabled
            or config.virtual_scroll is not None
            or config.stealth
            or bool(wait_for_selector)
            or (wait_until is not None and wait_until != "load")
        ):
            async with self._scraper_lock:
                return self._get_or_create_playwright(config)

        # 3. Default: HEAD-check first (no lock held), then pick the scraper.
        head_status = await self._head_check(url)

        async with self._scraper_lock:
            if head_status is not None and 200 <= head_status < 400:
                # Site responds well to simple HTTP → use httpx
                return self._get_or_create_httpx(config)

            if head_status is not None and head_status in (403, 429):
                # Site blocks simple HTTP clients — most often TLS/JA3 fingerprint
                # filtering. Try curl_cffi (Chrome impersonation) first: it clears
                # network-layer blocks far more cheaply than a browser. If it still
                # FAILs, scrape_url's AUTO fallback escalates to Playwright.
                logger.info(
                    f"HEAD returned {head_status} for {url}, trying CurlCffiScraper "
                    "(Chrome impersonation) before Playwright"
                )
                return self._get_or_create_curl_cffi(config)

            if head_status is None or head_status >= 500:
                # Timeout/error → try Playwright
                logger.info(f"HEAD failed/5xx for {url}, using PlaywrightScraper")
                return self._get_or_create_playwright(config)

            # Any other status → try httpx
            return self._get_or_create_httpx(config)

    async def scrape_url(
        self,
        url: str,
        scraper_type: ScraperType = ScraperType.AUTO,
        config: ScrapeConfig | None = None,
        **kwargs: Any,
    ) -> ScrapeResult:
        """Scrape a single URL and return extracted content.

        Args:
            url: The URL to scrape.
            scraper_type: Type of scraper to use.
            config: Optional scraping configuration.
            **kwargs: Additional scraper-specific options.

        Returns:
            ScrapeResult containing extracted content, metadata, and status.
        """
        logger.info(f"Scraping single URL: {url} with {scraper_type.value}")

        if config is None:
            config = ScrapeConfig(
                delay=settings.scraping_default_delay,
                user_agent=settings.scraping_user_agent,
            )

        if kwargs.get("timeout"):
            config.timeout = kwargs["timeout"]
        if kwargs.get("screenshot"):
            config.screenshot = kwargs["screenshot"]
        if kwargs.get("headers"):
            config.headers = kwargs["headers"]
        if kwargs.get("cookies"):
            config.cookies = kwargs["cookies"]
        if kwargs.get("user_agent"):
            config.user_agent = kwargs["user_agent"]

        # UI-interactive auth (form/storage_state) requires Playwright regardless
        # of what the caller asked for. Bearer/basic are just headers → any scraper.
        if config.auth_config and config.auth_config.type in ("form", "storage_state"):
            scraper_type = ScraperType.PLAYWRIGHT

        scraper_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k
            in (
                "viewport_width",
                "viewport_height",
                "wait_until",
                "screenshot",
                "wait_for_selector",
                "download_images",
            )
        }
        # Pass the resolved per-request config explicitly so each scraper reads
        # request-local state (auth/cookies/headers/user_agent/delay/...) via
        # _resolve_config rather than the shared self.config, which a concurrent
        # request may have mutated on the singleton instance. ``timeout`` is kept
        # as a separate kwarg for back-compat (governs nav + overall budget).
        scraper_kwargs["config"] = config
        scraper_kwargs["timeout"] = config.timeout

        scraper_name = "unknown"
        # Bound concurrent scrapes (see self._concurrency). Acquire/release via
        # try/finally so a permit is always returned, even on the FAILED path.
        await self._concurrency.acquire()
        try:
            # SSRF guard: reject loopback / link-local / private / *.internal
            # BEFORE any network I/O (incl. the AUTO-mode HEAD pre-check below).
            # Run the (blocking) DNS resolution off the event loop. Raising here
            # is converted to a FAILED ScrapeResult by the handler below, which is
            # exactly the per-URL semantics bulk wants; the single endpoint
            # pre-validates in the router for a clean 400.
            if settings.ssrf_protection_enabled:
                await asyncio.to_thread(assert_allowed_url, url, settings.ssrf_allowed_hosts)

            if scraper_type == ScraperType.AUTO:
                # _select_auto_scraper acquires _scraper_lock itself (only around the
                # get-or-create, not the HEAD pre-check). Forward the browser-only
                # wait signals so a HEAD-200 page with wait_for_selector/non-default
                # wait_until is not silently routed to httpx (which drops the wait).
                scraper = await self._select_auto_scraper(
                    url,
                    config,
                    wait_for_selector=scraper_kwargs.get("wait_for_selector"),
                    wait_until=scraper_kwargs.get("wait_until"),
                )
            else:
                scraper = await self._get_scraper(scraper_type, config)

            scraper_name = scraper.name
            result = await scraper.scrape(url, **scraper_kwargs)

            # Fallback logic for AUTO mode. CurlCffiScraper subclasses HttpxScraper,
            # so the isinstance covers it too — a failed impersonation attempt (e.g.
            # a JS challenge curl_cffi can't solve) escalates to Playwright.
            if result.status == ScrapeStatus.FAILED and scraper_type == ScraperType.AUTO:
                if isinstance(scraper, HttpxScraper | ScrapyScraper):
                    logger.warning(
                        f"{scraper.name} failed for {url}, falling back to PlaywrightScraper"
                    )
                    async with self._scraper_lock:
                        fallback = self._get_or_create_playwright(config)
                    scraper_name = fallback.name
                    fallback_kwargs = {**scraper_kwargs, "is_fallback": True}
                    # A CurlCffiScraper failure is the strongest hostile-site
                    # signal we have: AUTO only picks it after a 403/429, so the
                    # site rejected a plain client AND a real Chrome TLS/JA3
                    # fingerprint. Escalate to a stealth-patched browser rather
                    # than a vanilla one that advertises navigator.webdriver.
                    # Copy the config — mutating it would leak stealth into the
                    # caller's object (and, for bulk, into every later URL).
                    if isinstance(scraper, CurlCffiScraper) and not config.stealth:
                        logger.info(f"escalating {url} to stealth-patched Playwright")
                        fallback_kwargs["config"] = replace(config, stealth=True)
                    result = await fallback.scrape(url, **fallback_kwargs)

            result = await _enrich_extraction(result)
            record_scrape(result.scraper_used, result.status.value, result.processing_time_seconds)
            return result

        except Exception as e:
            logger.error(f"Error scraping URL {url}: {e}", exc_info=True)
            record_scrape(scraper_name, ScrapeStatus.FAILED.value, 0.0)
            return ScrapeResult(
                url=url,
                status=ScrapeStatus.FAILED,
                error=f"{type(e).__name__}: {e}" if str(e) else repr(e),
                scraper_used=scraper_name,
            )
        finally:
            self._concurrency.release()

    async def scrape_bulk(
        self,
        urls: list[str],
        scraper_type: ScraperType = ScraperType.AUTO,
        config: ScrapeConfig | None = None,
        **kwargs: Any,
    ) -> list[ScrapeResult]:
        """Scrape multiple URLs sequentially.

        Form-auth bulk optimisation: login once on the first URL, then reuse the
        captured storage_state for the remaining URLs so we avoid N form logins
        against the target origin.
        """
        logger.info(f"Scraping {len(urls)} URLs with {scraper_type.value}")

        if (
            config is not None
            and config.auth_config is not None
            and config.auth_config.type == "form"
            and urls
        ):
            return await self._scrape_bulk_with_form_auth_reuse(
                urls, scraper_type=scraper_type, config=config, **kwargs
            )

        results: list[ScrapeResult] = []
        for url in urls:
            try:
                result = await self.scrape_url(
                    url, scraper_type=scraper_type, config=config, **kwargs
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Error scraping {url}: {e}")
                results.append(
                    ScrapeResult(
                        url=url,
                        status=ScrapeStatus.FAILED,
                        error=f"{type(e).__name__}: {e}" if str(e) else repr(e),
                        scraper_used="unknown",
                    )
                )

        return results

    async def _scrape_bulk_with_form_auth_reuse(
        self,
        urls: list[str],
        *,
        scraper_type: ScraperType,
        config: ScrapeConfig,
        **kwargs: Any,
    ) -> list[ScrapeResult]:
        """Login once on urls[0], then reuse storage_state for urls[1:]."""
        first = await self.scrape_url(urls[0], scraper_type=scraper_type, config=config, **kwargs)
        results: list[ScrapeResult] = [first]

        if first.status != ScrapeStatus.COMPLETED or not first.storage_state:
            error = "skipped: bulk auth login failed on first URL"
            for url in urls[1:]:
                results.append(
                    ScrapeResult(
                        url=url,
                        status=ScrapeStatus.FAILED,
                        error=error,
                        scraper_used=first.scraper_used,
                    )
                )
            return results

        reused_config = copy.copy(config)
        reused_config.auth_config = AuthConfig(
            type="storage_state", storage_state=first.storage_state
        )
        for url in urls[1:]:
            try:
                result = await self.scrape_url(
                    url, scraper_type=scraper_type, config=reused_config, **kwargs
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Error scraping {url}: {e}")
                results.append(
                    ScrapeResult(
                        url=url,
                        status=ScrapeStatus.FAILED,
                        error=f"{type(e).__name__}: {e}" if str(e) else repr(e),
                        scraper_used="unknown",
                    )
                )
        return results

    async def _get_scraper(
        self, scraper_type: ScraperType, config: ScrapeConfig | None = None
    ) -> BaseScraper:
        """Get or create a scraper instance for the specified type."""
        async with self._scraper_lock:
            if scraper_type == ScraperType.PLAYWRIGHT:
                return self._get_or_create_playwright(config)

            if scraper_type == ScraperType.SCRAPY:
                return self._get_or_create_scrapy(config)

            if scraper_type == ScraperType.HTTPX:
                return self._get_or_create_httpx(config)

            if scraper_type == ScraperType.CURL_CFFI:
                return self._get_or_create_curl_cffi(config)

            # AUTO: shouldn't reach here (handled in scrape_url)
            return self._get_or_create_httpx(config)

    # NOTE: these are pure instance caches — they NEVER reassign self.config on an
    # existing instance. The per-request config travels with each scrape() call
    # (passed as the ``config`` kwarg by scrape_url); mutating the shared instance
    # here would let a concurrent request stomp another request's auth/cookies.
    def _get_or_create_playwright(self, config: ScrapeConfig | None) -> PlaywrightScraper:
        if self._playwright_scraper is None:
            self._playwright_scraper = PlaywrightScraper(
                config, headless=settings.playwright_headless
            )
        return self._playwright_scraper

    def _get_or_create_httpx(self, config: ScrapeConfig | None) -> HttpxScraper:
        if self._httpx_scraper is None:
            self._httpx_scraper = HttpxScraper(config)
        return self._httpx_scraper

    def _get_or_create_scrapy(self, config: ScrapeConfig | None) -> ScrapyScraper:
        if self._scrapy_scraper is None:
            self._scrapy_scraper = ScrapyScraper(config)
        return self._scrapy_scraper

    def _get_or_create_curl_cffi(self, config: ScrapeConfig | None) -> CurlCffiScraper:
        if self._curl_cffi_scraper is None:
            self._curl_cffi_scraper = CurlCffiScraper(config)
        return self._curl_cffi_scraper

    async def close(self) -> None:
        """Close all scrapers and release resources."""
        logger.info("Closing all scrapers...")

        if self._playwright_scraper:
            await self._playwright_scraper.close()
            self._playwright_scraper = None

        if self._httpx_scraper:
            await self._httpx_scraper.close()
            self._httpx_scraper = None

        if self._scrapy_scraper:
            await self._scrapy_scraper.close()
            self._scrapy_scraper = None

        if self._curl_cffi_scraper:
            await self._curl_cffi_scraper.close()
            self._curl_cffi_scraper = None

        logger.info("All scrapers closed")


_scraper_manager: ScraperManager | None = None


def get_scraper_manager() -> ScraperManager:
    """Get the global scraper manager singleton instance."""
    global _scraper_manager
    if _scraper_manager is None:
        _scraper_manager = ScraperManager()
    return _scraper_manager


__all__ = ["ScraperManager", "get_scraper_manager"]
