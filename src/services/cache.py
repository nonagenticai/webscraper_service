"""In-memory result cache for single-URL scrapes.

A short-TTL cache (``cachetools.TTLCache``) keyed by the request shape, so a
repeated scrape of the same URL+options inside the TTL window returns the prior
ScrapeResponse without re-hitting the network — the common case in re-ingest and
bulk-overlap workloads the RAG consumers drive.

Deliberately conservative:
- Only COMPLETED responses are stored (never cache a transient failure).
- Authed scrapes (``auth_config`` set) are NEVER cached — no credential-bound
  cross-caller reuse.
- ``cache_ttl_seconds=0`` disables the cache entirely; per-request ``no_cache``
  bypasses it.
- In-memory + single-replica only. A multi-replica future would want Redis; the
  interface here (make_key/get/set) is the seam for that.
"""

import hashlib
import json
from typing import TYPE_CHECKING

from cachetools import TTLCache

from config import get_settings

if TYPE_CHECKING:
    from schemas.scrape import ScrapeRequest, ScrapeResponse

# Request fields that change the scraped output and therefore the cache identity.
# (Transport/cosmetic-only fields are excluded so trivially-different requests
# still share a cache entry.)
#
# ⚠️ This is an allowlist and ``model_dump(include=...)`` silently ignores names
# it does not recognise. A new output-affecting request field that is not added
# here does not fail — it returns a HIT built for a DIFFERENT request, which is
# a wrong answer reported as a success. ``_NON_KEY_FIELDS`` below records the
# deliberate exclusions, and a test asserts the two sets together cover every
# ScrapeRequest field, so adding a field forces the decision instead of
# defaulting to the unsafe branch.
#
# That guard immediately found three stale names here — ``follow_links``,
# ``max_depth`` and ``allowed_domains`` are BulkScrapeRequest fields, and only
# the single-URL path is ever cached, so they had never contributed to a key.
# They are removed; computed keys are unchanged.
_KEY_FIELDS = (
    "url",
    "scraper_type",
    "javascript_enabled",
    "screenshot",
    "download_images",
    "wait_until",
    "wait_for_selector",
    "viewport_width",
    "viewport_height",
    "user_agent",
    "headers",
    "cookies",
    "link_query",
    "sort_links_by_score",
    "extraction_schema",
    "virtual_scroll",
    "stealth",
    # `delay` is a settle wait the Playwright backend actually performs
    # (`page.wait_for_timeout(cfg.delay * 1000)`) before reading the DOM, so a
    # request that asks for 5s of settle and one that asks for none do NOT
    # describe the same page. Sharing an entry returned the un-settled DOM as a
    # fresh 200 for the request that paid for the wait.
    "delay",
)

# Fields deliberately EXCLUDED from the cache identity, with the reason.
_NON_KEY_FIELDS = frozenset(
    {
        # Never cached at all — `cacheable()` refuses authed requests.
        "auth_config",
        # Controls cache use, not output.
        "no_cache",
        # A different budget for the SAME work: a scrape that completes is the
        # same page whether the ceiling was 30s or 60s, and only COMPLETED
        # responses are ever stored, so a timeout-shortened partial can't be
        # the thing being shared.
        "timeout",
    }
)


class ResultCache:
    """TTL cache of ScrapeResponse keyed by request shape."""

    def __init__(self, ttl_seconds: int, max_size: int) -> None:
        self._enabled = ttl_seconds > 0 and max_size > 0
        self._cache: TTLCache[str, ScrapeResponse] | None = (
            TTLCache(maxsize=max_size, ttl=ttl_seconds) if self._enabled else None
        )

    @property
    def enabled(self) -> bool:
        return self._enabled

    def cacheable(self, request: ScrapeRequest) -> bool:
        """Only anonymous (no auth) requests that didn't opt out are cacheable."""
        return (
            self._enabled
            and not getattr(request, "no_cache", False)
            and request.auth_config is None
        )

    def make_key(self, request: ScrapeRequest) -> str:
        """Stable hash of the output-affecting request fields."""
        payload = request.model_dump(mode="json", include=set(_KEY_FIELDS))
        blob = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()

    def get(self, key: str) -> ScrapeResponse | None:
        if self._cache is None:
            return None
        return self._cache.get(key)

    def set(self, key: str, response: ScrapeResponse) -> None:
        if self._cache is None:
            return
        self._cache[key] = response

    def clear(self) -> None:
        if self._cache is not None:
            self._cache.clear()


_result_cache: ResultCache | None = None


def get_result_cache() -> ResultCache:
    """Get the process-wide result cache singleton."""
    global _result_cache
    if _result_cache is None:
        settings = get_settings()
        _result_cache = ResultCache(settings.cache_ttl_seconds, settings.cache_max_size)
    return _result_cache


__all__ = ["_KEY_FIELDS", "_NON_KEY_FIELDS", "ResultCache", "get_result_cache"]
