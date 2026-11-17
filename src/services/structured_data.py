"""Embedded structured-data extraction from page HTML.

Pulls machine-readable metadata that publishers embed for search engines and
social cards — JSON-LD (schema.org Article/Product/Organization/...), microdata,
OpenGraph and microformats — into a single dict. This is the highest-leverage
*metadata* win for the RAG/LLM consumers: author, publish date, canonical entity
type and product/article fields arrive parsed instead of buried in raw
``<meta>``/``<script>`` tags.

RDFa and Dublin Core are deliberately excluded: under ``uniform=True`` extruct
emits structural placeholders for them on pages that carry no such data (e.g.
``dublincore`` → ``[{"@context": {}, "elements": [], "terms": []}]``) and RDFa
largely re-reports the OpenGraph tags already captured — pure noise for these
consumers.

Best-effort and defensive: a malformed page must never break a scrape, so every
failure path returns ``{}`` (or drops the offending syntax) rather than raising.
The serialized payload is capped so a pathological page with megabytes of
embedded JSON-LD cannot bloat the scrape response.
"""

import json
import logging
import os
from typing import Any

import extruct

logger = logging.getLogger(__name__)

# Syntaxes extracted, richest/most-common first. ``uniform=True`` normalizes the
# per-syntax output shapes so consumers see a predictable structure. RDFa +
# Dublin Core are excluded on purpose (see module docstring — placeholder noise).
_SYNTAXES = ("json-ld", "microdata", "opengraph", "microformat")

# The two highest-value syntaxes, kept when the full set blows the size budget.
_PRIORITY_SYNTAXES = ("json-ld", "opengraph")

# Cap the serialized structured_data payload (~256 KB) so an outlier page can't
# bloat the response. Env-tunable for ops without a redeploy.
try:
    _MAX_BYTES = int(os.getenv("WEBSCRAPER_STRUCTURED_DATA_MAX_BYTES", "262144"))
except TypeError, ValueError:
    _MAX_BYTES = 262144


def _nonempty(data: dict[str, Any]) -> dict[str, Any]:
    """Drop syntaxes that produced nothing, keeping the payload lean."""
    return {key: value for key, value in data.items() if value}


def _within_budget(data: dict[str, Any]) -> bool:
    """True when the JSON serialization fits the size cap."""
    try:
        return len(json.dumps(data, default=str)) <= _MAX_BYTES
    except Exception:
        return False


def extract_structured_data(html: str, base_url: str | None = None) -> dict[str, Any]:
    """Extract embedded structured data from page HTML.

    Args:
        html: Raw page HTML.
        base_url: Page URL, used to resolve relative IRIs in microdata/RDFa.

    Returns:
        Dict keyed by syntax (``json-ld``, ``microdata``, ``opengraph``,
        ``microformat``) with only the non-empty syntaxes present. ``{}`` on
        empty input, extraction failure, or when the result cannot be brought
        within the size budget.
    """
    if not html:
        return {}

    try:
        data = extruct.extract(
            html,
            base_url=base_url,
            syntaxes=list(_SYNTAXES),
            uniform=True,
            errors="log",  # a single bad block is logged + skipped, not fatal
        )
    except Exception as e:  # extruct must never break a scrape
        logger.warning(f"structured-data extraction failed: {e}")
        return {}

    result = _nonempty(data)
    if not result:
        return {}

    if _within_budget(result):
        return result

    # Oversized: keep only the highest-value syntaxes; drop entirely if still over.
    trimmed = _nonempty({key: result.get(key) for key in _PRIORITY_SYNTAXES})
    if trimmed and _within_budget(trimmed):
        logger.info("structured_data exceeded budget; kept json-ld+opengraph only")
        return trimmed
    logger.info("structured_data exceeded budget; dropped")
    return {}
