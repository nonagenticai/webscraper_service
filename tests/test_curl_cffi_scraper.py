"""Tests for the curl_cffi (Chrome-impersonation) scraper.

The SSRF guard is patched with a deterministic, network-free stub so the tests
exercise the scraper's redirect-walk + body-cap + status handling without real
DNS. The blocked-redirect test uses a stub that raises for the metadata/internal
target, proving the per-hop guard aborts the chain.
"""

import contextlib
from unittest.mock import patch

import pytest

import services.curl_cffi_scraper as cc
from services.base_scraper import ScrapeConfig, ScrapeStatus
from services.url_guard import UrlNotAllowedError

_HTML = (
    b"<html><head><title>Hi</title></head>"
    b"<body><article><p>The genuine article body lives here and is plenty long "
    b"to clear the recall guard threshold for real main content.</p></article></body></html>"
)


class _FakeResp:
    """Minimal stand-in for a curl_cffi streamed Response."""

    def __init__(self, status_code, *, chunks=None, location=None, url="https://example.com/"):
        self.status_code = status_code
        self.headers = {"location": location} if location else {}
        self.url = url
        self.encoding = "utf-8"
        self._chunks = chunks if chunks is not None else [_HTML]
        self.closed = False

    async def aiter_content(self):
        for chunk in self._chunks:
            yield chunk

    async def aclose(self):
        self.closed = True


class _FakeSession:
    """Async-context session returning queued responses; records .get kwargs."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_a):
        return False

    async def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self._responses.pop(0)


@contextlib.contextmanager
def _patched(session, *, block_substr=None):
    """Patch the curl_cffi session factory + the SSRF guard (network-free).

    block_substr: if a hop URL contains this substring, the guard raises
    UrlNotAllowedError (simulating a blocked host) — else it passes.
    """

    def _guard(url, _hosts=()):
        if block_substr and block_substr in url:
            raise UrlNotAllowedError(f"URL not allowed: blocked {url}")

    with (
        patch.object(cc, "AsyncSession", return_value=session),
        patch.object(cc, "assert_allowed_url", _guard),
    ):
        yield


@pytest.fixture
def scraper():
    return cc.CurlCffiScraper(ScrapeConfig())


@pytest.mark.asyncio
async def test_scrape_success(scraper):
    session = _FakeSession([_FakeResp(200)])
    with _patched(session):
        result = await scraper.scrape("https://example.com")
    assert result.status == ScrapeStatus.COMPLETED
    assert result.scraper_used == "CurlCffiScraper"
    assert result.title == "Hi"
    assert "genuine article body" in result.text_content
    assert result.metadata["status_code"] == 200
    # impersonate=chrome + no auto-redirects (we walk them ourselves).
    assert session.calls[0][1]["impersonate"] == "chrome"
    assert session.calls[0][1]["allow_redirects"] is False


@pytest.mark.asyncio
async def test_http_error_status_returns_failed(scraper):
    """A 4xx/5xx final status → FAILED so AUTO escalates to Playwright."""
    session = _FakeSession([_FakeResp(403, chunks=[b""])])
    with _patched(session):
        result = await scraper.scrape("https://example.com/forbidden")
    assert result.status == ScrapeStatus.FAILED
    assert "403" in result.error


@pytest.mark.asyncio
async def test_redirect_is_followed(scraper):
    """A 302 is followed to its Location and the final body is returned."""
    session = _FakeSession(
        [
            _FakeResp(302, location="https://example.com/final", chunks=[b""]),
            _FakeResp(200, url="https://example.com/final"),
        ]
    )
    with _patched(session):
        result = await scraper.scrape("https://example.com/start")
    assert result.status == ScrapeStatus.COMPLETED
    assert "genuine article body" in result.text_content
    assert [c[0] for c in session.calls] == [
        "https://example.com/start",
        "https://example.com/final",
    ]


@pytest.mark.asyncio
async def test_redirect_to_blocked_host_is_aborted(scraper):
    """THE security test: a 302 to cloud-metadata/RFC1918 is blocked before the
    second hop is ever issued → FAILED, body never fetched."""
    session = _FakeSession(
        [
            _FakeResp(302, location="http://169.254.169.254/latest/meta-data/", chunks=[b""]),
            _FakeResp(200),  # must NEVER be reached
        ]
    )
    with _patched(session, block_substr="169.254.169.254"):
        result = await scraper.scrape("https://example.com/redirector")
    assert result.status == ScrapeStatus.FAILED
    assert "not allowed" in result.error.lower()
    # Only the first hop was issued; the blocked target was never fetched.
    assert len(session.calls) == 1


@pytest.mark.asyncio
async def test_initial_blocked_url_fails_fast(scraper):
    """A directly-blocked target raises before any request is issued."""
    session = _FakeSession([_FakeResp(200)])
    with _patched(session, block_substr="foo.internal"):
        result = await scraper.scrape("http://foo.internal/secret")
    assert result.status == ScrapeStatus.FAILED
    assert "not allowed" in result.error.lower()
    assert len(session.calls) == 0


@pytest.mark.asyncio
async def test_oversized_body_is_failed(scraper, monkeypatch):
    """A body exceeding max_page_bytes is aborted mid-stream → FAILED (no OOM)."""
    monkeypatch.setattr(cc.get_settings(), "max_page_bytes", 10, raising=False)
    big = _FakeResp(200, chunks=[b"x" * 6, b"y" * 6])  # 12 bytes > 10
    session = _FakeSession([big])
    with _patched(session):
        result = await scraper.scrape("https://example.com/huge")
    assert result.status == ScrapeStatus.FAILED
    assert "exceeded" in result.error


@pytest.mark.asyncio
async def test_too_many_redirects_is_failed(scraper):
    """An endless redirect loop is bounded → FAILED, not infinite."""
    loop = [_FakeResp(302, location="https://example.com/next", chunks=[b""]) for _ in range(20)]
    session = _FakeSession(loop)
    with _patched(session):
        result = await scraper.scrape("https://example.com/loop")
    assert result.status == ScrapeStatus.FAILED
    assert "redirect" in result.error.lower()


@pytest.mark.asyncio
async def test_outbound_proxy_threaded_to_session(scraper, monkeypatch):
    """settings.outbound_proxy reaches the curl_cffi session.get(proxy=...) call."""
    monkeypatch.setattr(
        cc.get_settings(), "outbound_proxy", "http://proxy.test:3128", raising=False
    )
    session = _FakeSession([_FakeResp(200)])
    with _patched(session):
        await scraper.scrape("https://example.com")
    assert session.calls[0][1]["proxy"] == "http://proxy.test:3128"


class TestSameOriginDefaultPort:
    """The default port must be normalised before the same-origin comparison.

    ``urlparse("https://h/x").port`` is None while ``urlparse("https://h:443/x").port``
    is 443, so an explicit-port spelling of the SAME origin compared unequal and the
    walker stripped Authorization/cookies on a genuinely same-origin redirect. That
    fails CLOSED (over-stripping — a reliability bug, not a credential leak), but the
    fix must not invert into under-stripping: a real cross-origin hop still loses them.
    """

    @staticmethod
    def _cfg():
        return ScrapeConfig(
            headers={"Authorization": "Bearer s3cret", "X-Trace": "keep-me"},
            cookies={"session": "abc"},
        )

    @pytest.mark.asyncio
    async def test_default_port_spelling_keeps_credentials(self):
        """https://host/x → https://host:443/x is SAME origin: creds survive."""
        session = _FakeSession(
            [
                _FakeResp(302, location="https://example.com:443/final", chunks=[b""]),
                _FakeResp(200, url="https://example.com:443/final"),
            ]
        )
        scraper = cc.CurlCffiScraper(self._cfg())
        with _patched(session):
            result = await scraper.scrape("https://example.com/start")

        assert result.status == ScrapeStatus.COMPLETED
        second_hop = session.calls[1][1]
        assert second_hop["headers"]["Authorization"] == "Bearer s3cret"
        assert second_hop["cookies"] == {"session": "abc"}

    @pytest.mark.asyncio
    async def test_explicit_port_origin_keeps_credentials_on_bare_hop(self):
        """The mirror image: http://host:80/x → http://host/x is also same origin."""
        session = _FakeSession(
            [
                _FakeResp(302, location="http://example.com/final", chunks=[b""]),
                _FakeResp(200, url="http://example.com/final"),
            ]
        )
        scraper = cc.CurlCffiScraper(self._cfg())
        with _patched(session):
            result = await scraper.scrape("http://example.com:80/start")

        assert result.status == ScrapeStatus.COMPLETED
        assert session.calls[1][1]["headers"]["Authorization"] == "Bearer s3cret"

    @pytest.mark.asyncio
    async def test_cross_origin_still_strips_credentials(self):
        """Guard against inverting the fix: a real cross-origin hop loses creds."""
        session = _FakeSession(
            [
                _FakeResp(302, location="https://evil.example.net/final", chunks=[b""]),
                _FakeResp(200, url="https://evil.example.net/final"),
            ]
        )
        scraper = cc.CurlCffiScraper(self._cfg())
        with _patched(session):
            result = await scraper.scrape("https://example.com/start")

        assert result.status == ScrapeStatus.COMPLETED
        second_hop = session.calls[1][1]
        assert "Authorization" not in second_hop["headers"]
        assert second_hop["cookies"] is None
        # Non-credential headers still ride along.
        assert second_hop["headers"]["X-Trace"] == "keep-me"

    @pytest.mark.asyncio
    async def test_non_default_port_change_is_cross_origin(self):
        """:8443 is NOT the https default — that hop must still strip creds."""
        session = _FakeSession(
            [
                _FakeResp(302, location="https://example.com:8443/final", chunks=[b""]),
                _FakeResp(200, url="https://example.com:8443/final"),
            ]
        )
        scraper = cc.CurlCffiScraper(self._cfg())
        with _patched(session):
            result = await scraper.scrape("https://example.com/start")

        assert result.status == ScrapeStatus.COMPLETED
        assert "Authorization" not in session.calls[1][1]["headers"]

    @pytest.mark.asyncio
    async def test_scheme_change_is_cross_origin(self):
        """https→http on the same host is a downgrade, not same origin."""
        session = _FakeSession(
            [
                _FakeResp(302, location="http://example.com/final", chunks=[b""]),
                _FakeResp(200, url="http://example.com/final"),
            ]
        )
        scraper = cc.CurlCffiScraper(self._cfg())
        with _patched(session):
            result = await scraper.scrape("https://example.com/start")

        assert result.status == ScrapeStatus.COMPLETED
        assert "Authorization" not in session.calls[1][1]["headers"]


def test_url_guard_is_value_error():
    """Sanity: the guard error is a ValueError subclass (callers may catch broad)."""
    assert isinstance(UrlNotAllowedError("x"), ValueError)


class TestOutOfRangePort:
    """An out-of-range port must read as a blocked URL, not a raw ValueError.

    ``urlparse`` defers the 0-65535 check until ``.port`` is READ, and the walker
    reads it in ``_origin_of`` — for the caller's URL BEFORE the first guard call,
    and for every hop target. So ``https://h:99999/`` escaped through the generic
    ``except Exception`` as the stringified ``ValueError: Port out of range``,
    where a blocked/invalid-URL outcome belongs.

    The SSRF guard is patched out in every test here on purpose: ``_origin_of``
    must be safe on its own, since it also runs when ``ssrf_protection_enabled``
    is False.
    """

    def test_origin_of_rejects_an_out_of_range_port(self):
        with pytest.raises(UrlNotAllowedError, match="invalid port"):
            cc._origin_of("https://example.com:99999/x")

    def test_origin_of_still_normalises_a_valid_port(self):
        assert cc._origin_of("https://example.com/x") == cc._origin_of("https://example.com:443/x")

    @pytest.mark.asyncio
    async def test_caller_url_with_bad_port_fails_cleanly(self, scraper):
        session = _FakeSession([_FakeResp(200)])
        with _patched(session):
            result = await scraper.scrape("https://example.com:99999/start")

        assert result.status == ScrapeStatus.FAILED
        assert "invalid port" in result.error
        assert "ValueError" not in result.error
        assert session.calls == []  # rejected before any network I/O

    @pytest.mark.asyncio
    async def test_redirect_to_bad_port_fails_cleanly(self, scraper):
        session = _FakeSession(
            [
                _FakeResp(302, location="https://example.com:99999/final", chunks=[b""]),
                _FakeResp(200, url="https://example.com:99999/final"),
            ]
        )
        with _patched(session):
            result = await scraper.scrape("https://example.com/start")

        assert result.status == ScrapeStatus.FAILED
        assert "invalid port" in result.error
        assert "ValueError" not in result.error
        assert len(session.calls) == 1  # the bad hop was never issued
