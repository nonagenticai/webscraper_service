"""Tests for the in-memory result cache (services.cache)."""

from schemas.scrape import AuthConfig, AuthType, ScrapeRequest, ScrapeResponse, ScrapeStatus
from services.cache import _KEY_FIELDS, _NON_KEY_FIELDS, ResultCache


def _req(**kw) -> ScrapeRequest:
    return ScrapeRequest(url="https://example.com", **kw)


def _resp(status=ScrapeStatus.COMPLETED) -> ScrapeResponse:
    return ScrapeResponse(
        url="https://example.com",
        status=status,
        scraper_used="HttpxScraper",
        processing_time_seconds=0.1,
    )


def test_disabled_when_ttl_zero():
    cache = ResultCache(ttl_seconds=0, max_size=128)
    assert cache.enabled is False
    assert cache.cacheable(_req()) is False
    cache.set("k", _resp())  # no-op, must not raise
    assert cache.get("k") is None


def test_get_set_roundtrip():
    cache = ResultCache(ttl_seconds=300, max_size=128)
    key = cache.make_key(_req())
    assert cache.get(key) is None
    resp = _resp()
    cache.set(key, resp)
    assert cache.get(key) is resp


def test_key_is_stable_and_field_sensitive():
    cache = ResultCache(ttl_seconds=300, max_size=128)
    k1 = cache.make_key(_req())
    k2 = cache.make_key(_req())
    assert k1 == k2  # same request → same key
    # An output-affecting field changes the key.
    assert cache.make_key(_req(screenshot=True)) != k1
    assert cache.make_key(_req(scraper_type="playwright")) != k1
    assert cache.make_key(ScrapeRequest(url="https://other.example.com")) != k1


def test_cosmetic_fields_do_not_change_key():
    cache = ResultCache(ttl_seconds=300, max_size=128)
    # timeout is a ceiling on the SAME work, and only COMPLETED responses are
    # stored, so two requests that both finished describe the same page.
    cache_key = cache.make_key(_req(timeout=10.0))
    assert cache_key == cache.make_key(_req(timeout=200.0))


def test_delay_changes_the_key():
    """`delay` is a settle wait the browser actually performs before reading
    the DOM, so it changes what is scraped.

    Treating it as cosmetic returned the un-settled page — captured for a
    request that asked for no wait — as a fresh 200 to a caller who explicitly
    paid for five seconds of JS settling.
    """
    cache = ResultCache(ttl_seconds=300, max_size=128)
    assert cache.make_key(_req(delay=1.0)) != cache.make_key(_req(delay=5.0))


def test_no_cache_request_not_cacheable():
    cache = ResultCache(ttl_seconds=300, max_size=128)
    assert cache.cacheable(_req()) is True
    assert cache.cacheable(_req(no_cache=True)) is False


def test_authed_request_not_cacheable():
    """A request carrying auth_config must never be cached (no cross-caller reuse)."""
    cache = ResultCache(ttl_seconds=300, max_size=128)
    authed = _req(auth_config=AuthConfig(type=AuthType.BEARER, credential_id="cred-123"))
    assert cache.cacheable(authed) is False


def test_size_bounded_eviction():
    """The cache is size-bounded (memory safety on a small pod): exceeding
    max_size evicts an older entry rather than growing without limit."""
    cache = ResultCache(ttl_seconds=300, max_size=1)
    k1 = cache.make_key(_req(scraper_type="httpx"))
    k2 = cache.make_key(_req(scraper_type="playwright"))
    cache.set(k1, _resp())
    cache.set(k2, _resp())  # exceeds max_size=1 → k1 evicted
    assert len(cache._cache) == 1
    assert cache.get(k2) is not None


def test_clear():
    cache = ResultCache(ttl_seconds=300, max_size=128)
    key = cache.make_key(_req())
    cache.set(key, _resp())
    cache.clear()
    assert cache.get(key) is None


def test_key_fields_cover_every_request_field():
    """Every ScrapeRequest field must be a declared cache key or a declared exclusion.

    ``make_key`` uses ``model_dump(include=_KEY_FIELDS)``, which SILENTLY ignores
    names it does not know. A new output-affecting field that nobody adds to
    _KEY_FIELDS therefore does not fail loudly — it returns a cache HIT that was
    built for a different request. This test makes that drift impossible: adding
    a field to ScrapeRequest breaks here until it is classified.
    """
    classified = set(_KEY_FIELDS) | set(_NON_KEY_FIELDS)
    actual = set(ScrapeRequest.model_fields)
    assert actual - classified == set(), (
        f"unclassified ScrapeRequest field(s): {sorted(actual - classified)} — "
        "add to _KEY_FIELDS if they change the scraped output, else _NON_KEY_FIELDS"
    )
    assert classified - actual == set(), (
        f"stale cache-key name(s) not on ScrapeRequest: {sorted(classified - actual)}"
    )


def test_link_query_changes_the_cache_identity():
    """Two link_query values must not share an entry — the scores differ."""
    cache = ResultCache(ttl_seconds=300, max_size=128)
    assert cache.make_key(_req(link_query="pricing")) != cache.make_key(_req(link_query="careers"))
    assert cache.make_key(_req(link_query="pricing")) != cache.make_key(_req())


def test_sort_links_by_score_changes_the_cache_identity():
    cache = ResultCache(ttl_seconds=300, max_size=128)
    assert cache.make_key(_req(sort_links_by_score=True)) != cache.make_key(_req())
