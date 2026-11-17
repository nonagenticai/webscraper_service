"""SSRF guard for outbound scrape targets.

A scraper that authenticated callers can point at arbitrary URLs is a
server-side request forgery (SSRF) vector: without a guard a caller could
make the service fetch loopback, cloud-metadata (169.254.169.254),
RFC1918, or internal (``*.internal``) hosts it should never reach.

``assert_allowed_url`` is the single chokepoint, called by ScraperManager
before any network I/O (including the AUTO-mode HEAD pre-check) and per-URL
in bulk. It resolves the hostname and rejects the request if ANY resolved
address is loopback / link-local / private / reserved / multicast /
unspecified, or if the host is ``localhost`` / ``*.internal``.

Limitation (documented, accepted): the guard resolves DNS once here, but the
scrapers re-resolve when they fetch, so a TOCTOU / DNS-rebinding window
exists. Closing it fully would require pinning the resolved IP into every
scraper's transport, which is out of scope for this guard. This still blocks
the overwhelming majority of SSRF attempts (static internal hostnames,
literal private IPs, metadata endpoints).
"""

import asyncio
import ipaddress
import logging
import socket
from collections.abc import Callable, Collection, Coroutine
from typing import Any, Protocol
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

_ALLOWED_SCHEMES = frozenset({"http", "https"})
# Hostnames that always resolve to / mean the local or internal network,
# independent of (or to short-circuit) DNS resolution.
_BLOCKED_HOST_SUFFIXES = (".internal",)
_BLOCKED_HOST_EXACT = frozenset({"localhost"})


class UrlNotAllowedError(ValueError):
    """Raised when a target URL is disallowed by the SSRF guard."""


def _ip_is_blocked(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """True if an IP belongs to a range a public scraper must never reach.

    ``is_global`` is the load-bearing check, not the flag list. The named flags
    below miss RFC 6598 shared address space (100.64.0.0/10) — which on this
    fleet is the TAILSCALE TAILNET: the k3s control plane (100.64.0.10), the
    docker registry (100.64.0.11), the GPU hosts. Python reports those as
    ``is_private=False, is_reserved=False``, so a caller could point a scrape at
    the kube API or the registry and the guard would wave it through. Every one
    of them is ``is_global=False``, as is every other range a scraper has no
    business reaching, so that is the rule and the flags are the explanation.
    """
    # An IPv4-mapped IPv6 literal (::ffff:100.64.0.1) must be judged on the
    # address it actually reaches, not on the wrapper.
    mapped = getattr(ip, "ipv4_mapped", None)
    if mapped is not None:
        ip = mapped

    return (
        not ip.is_global
        or ip.is_loopback
        or ip.is_link_local  # 169.254.0.0/16 + fe80::/10 (incl. cloud metadata)
        or ip.is_private  # RFC1918 + unique-local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


def assert_allowed_url(url: str, allowed_hosts: Collection[str] = ()) -> None:
    """Validate a scrape target, raising UrlNotAllowedError if disallowed.

    Args:
        url: The target URL to validate.
        allowed_hosts: Exact hostnames/IPs that bypass the IP-range checks
            (still required to use an http(s) scheme). Lets ops permit a
            specific trusted internal host (or a loopback test server).

    Raises:
        UrlNotAllowedError: scheme not http(s); host missing; port outside
            0-65535; host is localhost / ``*.internal``; DNS fails
            (fail-closed); or any resolved IP is loopback/link-local/private/
            reserved/multicast/unspecified.
    """
    try:
        parsed = urlparse(url)
    except Exception as e:  # pragma: no cover — urlparse is very tolerant
        raise UrlNotAllowedError(f"URL not allowed: unparseable ({e})") from e

    scheme = (parsed.scheme or "").lower()
    if scheme not in _ALLOWED_SCHEMES:
        raise UrlNotAllowedError(f"URL not allowed: scheme {scheme or '(none)'!r} (use http/https)")

    host = (parsed.hostname or "").lower()
    if not host:
        raise UrlNotAllowedError("URL not allowed: missing host")

    # `urlparse` accepts an out-of-range port lazily: it only raises when `.port`
    # is READ. Since this guard never read it, `https://h:99999/` sailed through
    # as ALLOWED, and the ValueError surfaced later — from whichever downstream
    # caller happened to touch `.port` first — as a raw stringified ValueError
    # where a "blocked/invalid URL" outcome belongs. Force the parse here so an
    # unusable authority is rejected by the chokepoint, like any other bad URL.
    try:
        parsed.port
    except ValueError as e:
        raise UrlNotAllowedError(f"URL not allowed: invalid port ({e})") from e

    # A fully-qualified name may carry a trailing root dot, and DNS resolves it
    # identically — but `"webscraper.example.com.".endswith(".internal")` is False,
    # so the suffix rules below were one character away from being bypassed.
    host = host.rstrip(".")
    if not host:
        raise UrlNotAllowedError("URL not allowed: missing host")

    # Explicit operator allowlist bypasses the range checks (scheme already ok).
    if host in {h.lower() for h in allowed_hosts}:
        return

    if host in _BLOCKED_HOST_EXACT or host.endswith(_BLOCKED_HOST_SUFFIXES):
        raise UrlNotAllowedError(f"URL not allowed: internal host {host!r}")

    # A literal IP in the URL still has to clear the IP filter.
    try:
        literal_ip = ipaddress.ip_address(host)
    except ValueError:
        literal_ip = None
    if literal_ip is not None:
        if _ip_is_blocked(literal_ip):
            raise UrlNotAllowedError(f"URL not allowed: blocked address {host}")
        return

    # Resolve the hostname; reject if ANY address is in a blocked range.
    try:
        infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    except socket.gaierror as e:
        # Fail closed: a host we can't resolve is not a host we should fetch.
        raise UrlNotAllowedError(f"URL not allowed: cannot resolve host {host!r}") from e

    resolved = {info[4][0] for info in infos}
    for addr in resolved:
        try:
            ip = ipaddress.ip_address(addr)
        except ValueError:  # pragma: no cover — getaddrinfo returns valid IPs
            continue
        if _ip_is_blocked(ip):
            raise UrlNotAllowedError(
                f"URL not allowed: host {host!r} resolves to blocked address {addr}"
            )


class _HasUrl(Protocol):
    """Minimal duck-type for an httpx.Request (avoids importing httpx here)."""

    @property
    def url(self) -> object: ...


def make_outbound_guard_hook(
    allowed_hosts: Collection[str] = (),
) -> Callable[[_HasUrl], Coroutine[Any, Any, None]]:
    """Build an httpx ``request`` event hook that SSRF-guards EVERY hop.

    The chokepoint ``assert_allowed_url`` only validates the *initial* URL, but
    httpx (and the image-download client) follow redirects, so a public page
    that ``302``s to ``169.254.169.254`` / loopback / RFC1918 / ``*.internal``
    would be fetched *after* the guard passed. httpx fires this hook for the
    initial request AND every redirect target; raising ``UrlNotAllowedError``
    aborts the whole chain. Blocking DNS resolution runs off the event loop.

    Attach via ``httpx.AsyncClient(event_hooks={"request": [hook]})``.
    """
    hosts = tuple(allowed_hosts)

    async def _hook(request: _HasUrl) -> None:
        await asyncio.to_thread(assert_allowed_url, str(request.url), hosts)

    return _hook


__all__ = ["UrlNotAllowedError", "assert_allowed_url", "make_outbound_guard_hook"]
