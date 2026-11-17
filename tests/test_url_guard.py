"""Tests for the SSRF URL guard."""

from unittest.mock import patch

import pytest

from services.url_guard import (
    UrlNotAllowedError,
    assert_allowed_url,
    make_outbound_guard_hook,
)


def _addrinfo(*ips: str):
    """Build a getaddrinfo-shaped return value for the given IP strings."""
    return [(2, 1, 6, "", (ip, 0)) for ip in ips]


class TestSchemeAndHost:
    @pytest.mark.parametrize("url", ["ftp://example.com", "file:///etc/passwd", "gopher://x"])
    def test_non_http_scheme_rejected(self, url):
        with pytest.raises(UrlNotAllowedError, match="scheme"):
            assert_allowed_url(url)

    def test_missing_host_rejected(self):
        with pytest.raises(UrlNotAllowedError, match="missing host"):
            assert_allowed_url("http:///path-only")


class TestBlockedHostnames:
    def test_localhost_rejected(self):
        with pytest.raises(UrlNotAllowedError, match="internal host"):
            assert_allowed_url("http://localhost:8080/x")

    @pytest.mark.parametrize(
        "url",
        [
            "http://registry.example.com/v2/",
            "https://postgres.example.com:5432",
            "http://anything.INTERNAL/",
        ],
    )
    def test_internal_suffix_rejected(self, url):
        with pytest.raises(UrlNotAllowedError, match="internal host"):
            assert_allowed_url(url)


class TestBlockedLiteralIPs:
    @pytest.mark.parametrize(
        "url",
        [
            "http://127.0.0.1/",  # loopback
            "http://169.254.169.254/latest/meta-data/",  # cloud metadata (link-local)
            "http://10.0.0.1/",  # RFC1918
            "http://192.168.1.1/",  # RFC1918
            "http://172.16.5.5/",  # RFC1918
            "http://[::1]/",  # IPv6 loopback
            "http://0.0.0.0/",  # unspecified
        ],
    )
    def test_blocked_ip_literals(self, url):
        with pytest.raises(UrlNotAllowedError, match="blocked address"):
            assert_allowed_url(url)

    def test_public_ip_literal_allowed(self):
        # 93.184.216.34 (example.com); literal public IP → no DNS, allowed.
        assert_allowed_url("https://93.184.216.34/") is None


class TestDnsResolution:
    def test_public_host_allowed(self):
        with patch(
            "services.url_guard.socket.getaddrinfo", return_value=_addrinfo("93.184.216.34")
        ):
            assert assert_allowed_url("https://example.com/page") is None

    def test_host_resolving_to_private_rejected(self):
        # DNS-rebinding-style: a public name that resolves to a private IP.
        with patch("services.url_guard.socket.getaddrinfo", return_value=_addrinfo("10.1.2.3")):
            with pytest.raises(UrlNotAllowedError, match="resolves to blocked address"):
                assert_allowed_url("https://sneaky.example.com/")

    def test_any_resolved_private_ip_rejected(self):
        # Mixed result set: one public, one private → reject (fail safe).
        with patch(
            "services.url_guard.socket.getaddrinfo",
            return_value=_addrinfo("93.184.216.34", "127.0.0.1"),
        ):
            with pytest.raises(UrlNotAllowedError):
                assert_allowed_url("https://mixed.example.com/")

    def test_unresolvable_host_fails_closed(self):
        import socket as _socket

        with patch("services.url_guard.socket.getaddrinfo", side_effect=_socket.gaierror(8, "no")):
            with pytest.raises(UrlNotAllowedError, match="cannot resolve"):
                assert_allowed_url("https://does-not-exist.invalid/")


class TestAllowlist:
    def test_allowlisted_host_bypasses_ip_check(self):
        # 127.0.0.1 normally blocked, but an explicit allowlist permits it
        # (used by the loopback test server / trusted internal hosts).
        assert assert_allowed_url("http://127.0.0.1:9000/x", allowed_hosts=["127.0.0.1"]) is None

    def test_allowlist_is_case_insensitive_for_hostnames(self):
        assert assert_allowed_url("http://My.Internal/x", allowed_hosts=["my.internal"]) is None

    def test_allowlist_still_requires_http_scheme(self):
        with pytest.raises(UrlNotAllowedError, match="scheme"):
            assert_allowed_url("ftp://127.0.0.1/x", allowed_hosts=["127.0.0.1"])


class _FakeRequest:
    """Minimal stand-in for an httpx.Request (only ``.url`` is read)."""

    def __init__(self, url: str) -> None:
        self.url = url


class TestOutboundGuardHook:
    """The per-hop event hook that re-validates each redirect target."""

    @pytest.mark.asyncio
    async def test_hook_blocks_metadata_redirect_target(self):
        hook = make_outbound_guard_hook()
        with pytest.raises(UrlNotAllowedError, match="blocked address"):
            await hook(_FakeRequest("http://169.254.169.254/latest/meta-data/"))

    @pytest.mark.asyncio
    async def test_hook_blocks_internal_suffix(self):
        hook = make_outbound_guard_hook()
        with pytest.raises(UrlNotAllowedError, match="internal host"):
            await hook(_FakeRequest("http://registry.example.com/v2/"))

    @pytest.mark.asyncio
    async def test_hook_allows_public_literal(self):
        hook = make_outbound_guard_hook()
        assert await hook(_FakeRequest("https://93.184.216.34/page")) is None

    @pytest.mark.asyncio
    async def test_hook_honors_allowed_hosts(self):
        hook = make_outbound_guard_hook(allowed_hosts=["127.0.0.1"])
        assert await hook(_FakeRequest("http://127.0.0.1:9000/x")) is None


class TestTailnetAndSuffixBypasses:
    """Ranges and host forms that the flag-list rules missed."""

    @pytest.mark.parametrize(
        "addr",
        [
            "100.64.0.1",  # RFC 6598 shared address space, lower bound
            "100.64.0.10",  # k3s control plane on this fleet
            "100.64.0.11",  # docker registry
            "100.127.255.254",  # upper reach of 100.64.0.0/10
        ],
    )
    def test_cgnat_tailnet_range_is_blocked(self, addr):
        """The tailnet is where every piece of this fleet's infrastructure lives.

        Python reports 100.64.0.0/10 as is_private=False and is_reserved=False,
        so a flag-list guard waved it straight through — a caller could aim a
        scrape at the kube API or the registry.
        """
        with pytest.raises(UrlNotAllowedError):
            assert_allowed_url(f"http://{addr}/")

    def test_ipv4_mapped_ipv6_literal_is_judged_on_the_mapped_address(self):
        with pytest.raises(UrlNotAllowedError):
            assert_allowed_url("http://[::ffff:100.64.0.1]/")

    def test_trailing_dot_does_not_bypass_the_internal_suffix_rule(self):
        """`"x.internal.".endswith(".internal")` is False, but DNS resolves both."""
        with pytest.raises(UrlNotAllowedError):
            assert_allowed_url("https://webscraper.example.com./api/scrape/url")

    def test_trailing_dot_does_not_bypass_localhost(self):
        with pytest.raises(UrlNotAllowedError):
            assert_allowed_url("http://localhost./")


class TestInvalidPort:
    """An out-of-range port is an unusable URL, not an ALLOWED one.

    `urlparse` defers the range check until `.port` is READ. The guard never read
    it, so `https://h:99999/` returned ALLOWED and the ValueError surfaced later
    from whichever caller touched `.port` first — as a raw stringified ValueError
    where a blocked/invalid-URL outcome belongs.
    """

    @pytest.mark.parametrize("url", ["https://example.com:99999/", "http://example.com:65536/x"])
    def test_out_of_range_port_rejected(self, url):
        with patch("socket.getaddrinfo", return_value=_addrinfo("93.184.216.34")):
            with pytest.raises(UrlNotAllowedError, match="invalid port"):
                assert_allowed_url(url)

    def test_out_of_range_port_rejected_before_dns(self):
        """The rejection must not depend on the host resolving."""
        with patch("socket.getaddrinfo", side_effect=AssertionError("must not resolve")):
            with pytest.raises(UrlNotAllowedError, match="invalid port"):
                assert_allowed_url("https://example.com:99999/")

    def test_out_of_range_port_not_rescued_by_the_allowlist(self):
        """An operator allowlist bypasses IP-range checks, never URL validity."""
        with pytest.raises(UrlNotAllowedError, match="invalid port"):
            assert_allowed_url("https://example.com:99999/", allowed_hosts=["example.com"])

    def test_in_range_port_still_allowed(self):
        with patch("socket.getaddrinfo", return_value=_addrinfo("93.184.216.34")):
            assert_allowed_url("https://example.com:8443/x")
