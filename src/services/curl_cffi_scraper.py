"""Browser-impersonation scraper using curl_cffi (TLS/JA3 fingerprint mimicry).

Many sites fingerprint the TLS/JA3 + HTTP/2 handshake and block plain HTTP
clients (httpx) with 403/429 even though the page is static HTML. curl_cffi
impersonates a real Chrome handshake, getting past that network-layer block at a
tiny fraction of what spinning up headless Chromium costs. It does NOT execute
JavaScript, so a page behind a JS challenge ("Checking your browser…") still
needs the Playwright fallback.

Reuses ``HttpxScraper``'s post-fetch assembly (parse → text/metadata/links/
images; markdown + structured-data are added centrally by ScraperManager) so the
output shape is identical across backends — only the fetch transport differs.

SSRF: curl_cffi follows redirects internally with no per-hop hook, so we fetch
with ``allow_redirects=False`` and walk the chain ourselves, re-validating EVERY
hop through ``assert_allowed_url`` before issuing it. A public page that ``302``s
to loopback / ``169.254.169.254`` / RFC1918 / ``*.internal`` is aborted (FAILED),
matching the httpx event-hook guard. The body is stream-capped at
``max_page_bytes`` exactly like httpx so an oversized body can't OOM the pod.
"""

import asyncio
import logging
import time
from typing import Any
from urllib.parse import urljoin, urlparse

from curl_cffi.requests import AsyncSession

from config import Settings, get_settings
from services.base_scraper import ScrapeConfig, ScrapeResult, ScrapeStatus
from services.httpx_scraper import HttpxScraper, ResponseTooLargeError
from services.url_guard import UrlNotAllowedError, assert_allowed_url

logger = logging.getLogger(__name__)

# Browser profile curl_cffi impersonates. "chrome" tracks the latest bundled
# Chrome fingerprint, so it stays current as curl_cffi is upgraded.
_IMPERSONATE = "chrome"
_REDIRECT_STATUS = frozenset({301, 302, 303, 307, 308})
_MAX_REDIRECTS = 10

# Request headers that carry caller credentials and must not cross an origin.
_CREDENTIAL_HEADERS = frozenset({"authorization", "cookie", "proxy-authorization"})

# Default ports per scheme, so "https://h/x" and "https://h:443/x" compare equal.
_DEFAULT_PORTS: dict[str, int] = {"http": 80, "https": 443}


def _origin_of(url: str) -> tuple[str, str, int | None]:
    """(scheme, host, port) — the tuple that decides same-origin for credentials.

    The port is normalised to the scheme default, because ``urlparse`` reports
    ``None`` for ``https://host/x`` but ``443`` for ``https://host:443/x``. Comparing
    the raw ports made two spellings of the SAME origin look different, so a
    genuinely same-origin redirect had its Authorization header and cookies stripped.
    A non-default port (``:8443``) or a scheme change still reads as cross-origin.

    Raises:
        UrlNotAllowedError: the authority carries an out-of-range port. ``urlparse``
            defers that error until ``.port`` is read, and the caller's origin is
            computed HERE — before the SSRF guard runs on the first hop — so a
            ``https://h:99999/`` target used to escape as a raw stringified
            ``ValueError`` from the generic handler. It is an unusable URL, so it
            gets the same blocked-URL outcome as any other rejected target.
    """
    parsed = urlparse(url)
    scheme = (parsed.scheme or "").lower()
    try:
        explicit_port = parsed.port
    except ValueError as e:
        raise UrlNotAllowedError(f"URL not allowed: invalid port ({e})") from e
    port = explicit_port if explicit_port is not None else _DEFAULT_PORTS.get(scheme)
    return (scheme, (parsed.hostname or "").lower(), port)


def _without_credentials(headers: dict[str, str]) -> dict[str, str]:
    """Drop credential-bearing headers before following a cross-origin redirect."""
    return {k: v for k, v in headers.items() if k.lower() not in _CREDENTIAL_HEADERS}


class CurlCffiScraper(HttpxScraper):
    """Static-HTML scraper that impersonates a Chrome TLS fingerprint.

    Subclasses ``HttpxScraper`` purely to reuse the backend-agnostic post-fetch
    assembly (``_assemble_result`` + the ``_extract_*`` helpers); the httpx
    transport methods (``_build_client``/``_get_with_retries``) are unused.
    """

    async def scrape(self, url: str, **kwargs: Any) -> ScrapeResult:
        """Scrape a URL via curl_cffi Chrome impersonation.

        Returns:
            ScrapeResult with extracted content (COMPLETED), or FAILED on an HTTP
            error status (so AUTO falls back to Playwright), an SSRF-blocked
            redirect, an oversized body, or any transport error.
        """
        start_time = time.time()
        cfg = self._resolve_config(kwargs)
        timeout = float(kwargs.get("timeout") or cfg.timeout)
        settings = get_settings()

        try:
            html_content, status_code, final_url = await self._fetch(url, cfg, timeout, settings)

            if status_code >= 400:
                # Mirror httpx's raise_for_status → FAILED so the AUTO fallback in
                # ScraperManager retries with Playwright.
                logger.info(f"curl_cffi got HTTP {status_code} for {url}")
                return self._failed(url, f"HTTP {status_code}", start_time)

            download_images: bool = kwargs.get("download_images", cfg.download_images)
            return await self._assemble_result(
                url,
                html_content,
                status_code,
                final_url,
                cfg,
                timeout,
                download_images,
                start_time,
            )

        except ResponseTooLargeError as e:
            logger.warning(f"Oversized response scraping {url}: {e}")
            return self._failed(url, str(e), start_time)
        except UrlNotAllowedError as e:
            # A redirect pointed at a blocked host — aborted before the fetch.
            logger.warning(f"SSRF guard blocked a redirect while scraping {url}: {e}")
            return self._failed(url, str(e), start_time)
        except Exception as e:
            logger.error(f"Error scraping {url} with curl_cffi: {e}", exc_info=True)
            error = f"{type(e).__name__}: {e}" if str(e) else repr(e)
            return self._failed(url, error, start_time)

    def _failed(self, url: str, error: str, start_time: float) -> ScrapeResult:
        return ScrapeResult(
            url=url,
            status=ScrapeStatus.FAILED,
            error=error,
            scraper_used=self.name,
            processing_time_seconds=time.time() - start_time,
        )

    async def _fetch(
        self, url: str, cfg: ScrapeConfig, timeout: float, settings: Settings
    ) -> tuple[str, int, str]:
        """Fetch ``url`` with manual, SSRF-guarded redirect following.

        Returns ``(html, status_code, final_url)``. Raises ``UrlNotAllowedError``
        if any hop targets a blocked host, ``ResponseTooLargeError`` past the body
        cap, or ``RuntimeError`` if the redirect chain is too long.
        """
        headers: dict[str, str] = {}
        if cfg.user_agent:
            headers["User-Agent"] = cfg.user_agent
        if cfg.headers:
            headers.update(cfg.headers)
        proxy = settings.outbound_proxy or None
        max_bytes = settings.max_page_bytes

        current = url
        origin = _origin_of(url)
        async with AsyncSession() as session:
            for _ in range(_MAX_REDIRECTS + 1):
                # SSRF: re-validate the host we're about to hit BEFORE issuing it.
                if settings.ssrf_protection_enabled:
                    await asyncio.to_thread(
                        assert_allowed_url, current, settings.ssrf_allowed_hosts
                    )

                # Credentials are scoped to the origin the caller aimed at. This
                # walker is hand-rolled (allow_redirects=False), so nothing was
                # stripping them: a 302 to an attacker-controlled host replayed
                # the caller's Authorization header and cookies straight to it.
                # httpx and requests both drop auth on cross-origin redirects;
                # so does this now.
                same_origin = _origin_of(current) == origin
                hop_headers = headers if same_origin else _without_credentials(headers)
                hop_cookies = (cfg.cookies or None) if same_origin else None

                response = await session.get(
                    current,
                    impersonate=_IMPERSONATE,
                    allow_redirects=False,
                    timeout=timeout,
                    headers=hop_headers or None,
                    cookies=hop_cookies,
                    proxy=proxy,
                    verify=True,
                    stream=True,
                )
                try:
                    status = response.status_code
                    location = response.headers.get("location")
                    if status in _REDIRECT_STATUS and location:
                        current = urljoin(current, location)
                        continue
                    html_content = await self._read_capped_curl(response, current, max_bytes)
                    final_url = str(response.url) or current
                    return html_content, status, final_url
                finally:
                    await response.aclose()

        raise RuntimeError(f"too many redirects (> {_MAX_REDIRECTS}) for {url}")

    @staticmethod
    async def _read_capped_curl(response: Any, url: str, max_bytes: int) -> str:
        """Stream a curl_cffi response body into text, aborting past ``max_bytes``.

        Mirrors ``HttpxScraper._read_capped`` so the new backend cannot reintroduce
        the unbounded-body OOM/DoS the httpx path was hardened against.
        """
        chunks: list[bytes] = []
        total = 0
        async for chunk in response.aiter_content():
            total += len(chunk)
            if total > max_bytes:
                raise ResponseTooLargeError(f"response body exceeded {max_bytes} bytes for {url}")
            chunks.append(chunk)
        body = b"".join(chunks)
        encoding = response.encoding or "utf-8"
        try:
            return body.decode(encoding, errors="replace")
        except LookupError, TypeError:
            # An unknown/invalid charset label — fall back to UTF-8.
            return body.decode("utf-8", errors="replace")

    async def close(self) -> None:
        """No persistent resources (sessions are per-request)."""
        logger.info("CurlCffiScraper closed")


__all__ = ["CurlCffiScraper"]
