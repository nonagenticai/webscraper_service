"""Prometheus metrics for scrape operations.

A tiny shared registry of metric objects so the manager (scrape outcomes) and the
scrape router (cache hits/misses) record into the same series, exposed by the
``/metrics`` router. The fleet already runs Grafana/Prometheus, so this slots into
existing observability with no new infrastructure.

Kept deliberately low-cardinality: labels are the backend name (a small fixed set
of scraper classes) and the scrape status — never the URL or any caller value.
"""

from prometheus_client import Counter, Histogram

SCRAPES_TOTAL = Counter(
    "webscraper_scrapes_total",
    "Total scrape operations by backend and final status.",
    ["backend", "status"],
)

SCRAPE_DURATION_SECONDS = Histogram(
    "webscraper_scrape_duration_seconds",
    "Scrape wall-clock duration in seconds by backend.",
    ["backend"],
)

CACHE_LOOKUPS_TOTAL = Counter(
    "webscraper_cache_lookups_total",
    "Result-cache lookups by outcome (hit / miss / bypass).",
    ["result"],
)


def record_scrape(backend: str | None, status: str, duration_seconds: float) -> None:
    """Record one completed scrape attempt (any final status)."""
    label = backend or "unknown"
    SCRAPES_TOTAL.labels(backend=label, status=status).inc()
    SCRAPE_DURATION_SECONDS.labels(backend=label).observe(max(duration_seconds, 0.0))


def record_cache(result: str) -> None:
    """Record a result-cache lookup outcome: 'hit', 'miss', or 'bypass'."""
    CACHE_LOOKUPS_TOTAL.labels(result=result).inc()


__all__ = [
    "CACHE_LOOKUPS_TOTAL",
    "SCRAPES_TOTAL",
    "SCRAPE_DURATION_SECONDS",
    "record_cache",
    "record_scrape",
]
