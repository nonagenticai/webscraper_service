"""Declarative CSS-selector extraction of repeated page structures.

Fills the one capability this service had no answer for: pulling *rows* out of a
page. A directory listing, a results table, a team page or a pricing grid all
carry N structurally identical records, and until now a caller got back
``html_content`` and had to re-parse it themselves. With an ``extraction_schema``
on the request they instead get ``extracted_data`` — a list of dicts shaped the
way they asked.

Backend-agnostic by construction: it runs against the ``html_content`` every
backend already produces, so httpx, curl_cffi, Playwright and Scrapy all support
it without touching routing.

Adapted from crawl4ai's ``JsonElementExtractionStrategy`` /
``JsonCssExtractionStrategy`` (``crawl4ai/extraction_strategy.py``) at commit
``7e80152``, © UncleCode, Apache-2.0 — see THIRD_PARTY_NOTICES.md. The schema
dialect is kept compatible (``baseSelector`` is accepted verbatim) so an existing
crawl4ai schema can be pasted straight into a request.

Deliberately NOT carried over
-----------------------------
``computed`` fields
    Upstream supports ``expression`` (evaluated) and ``function`` (a Python
    callable). Neither can cross an HTTP boundary safely or at all — upstream
    already disabled ``expression`` for exactly this reason. Omitted.

``regex`` fields
    A caller-supplied pattern run over page text is a ReDoS vector: Python's
    ``re`` has no timeout, this service runs a single replica, and the
    extraction is synchronous CPU work on the event loop — one catastrophic
    backtrack takes the pod down. Omitted rather than half-mitigated. Callers
    needing it can post-process ``text``/``html`` output.

``source`` (sibling traversal)
    Upstream's ``+selector`` next-sibling hop. Niche, and it complicates the
    selector contract. Omitted for now; revisit if a caller asks.

Everything the extraction produces is bounded — item count, per-value length and
schema size — because a hostile-or-clumsy selector against a large page is the
obvious denial-of-service here.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from bs4 import BeautifulSoup

if TYPE_CHECKING:
    from bs4.element import Tag

    from schemas.scrape import ExtractionField, ExtractionSchema

logger = logging.getLogger(__name__)

# Hard ceilings applied during extraction. The schema model bounds the *request*
# side (field count, nesting depth); these bound what a legitimate schema can
# pull off a hostile or simply enormous page.
MAX_EXTRACTED_ITEMS = 1_000
MAX_EXTRACTED_VALUE_LENGTH = 50_000
# Per-field fan-out. MAX_EXTRACTED_ITEMS bounds the number of RECORDS; without
# this a single `list` field whose selector matches every node on the page made
# each record unbounded, so the total was unbounded too.
MAX_EXTRACTED_LIST_ITEMS = 500


class ExtractionResult:
    """Extraction output plus what had to be given up to produce it.

    A bare ``list`` could not distinguish "the page has no rows", "your selector
    is invalid" and "there were 5,000 rows and you are looking at 1,000" — all
    three arrived as a successful-looking response.

    ``field_errors`` covers a fourth case, distinct from all three: a
    *syntactically valid* selector (invalid CSS is already rejected at request
    validation, see ``schemas.scrape._assert_selectors_compile``) that still
    raised while actually running against this page — a genuine bug or a
    pathological DOM, not "the page has no matching element". That is NOT the
    same thing as a selector correctly matching nothing, which stays a normal
    empty/default result and is never added here.
    """

    __slots__ = ("field_errors", "items", "truncated")

    def __init__(
        self,
        items: list[dict[str, Any]],
        *,
        truncated: bool = False,
        field_errors: list[str] | None = None,
    ) -> None:
        self.items = items
        self.truncated = truncated
        self.field_errors = field_errors or []


def extract_structured(html: str, schema: ExtractionSchema) -> ExtractionResult:
    """Apply a declarative CSS schema to page HTML.

    Selector validity is enforced when the request is parsed
    (``ExtractionSchema`` compiles every selector), so an invalid selector is a
    422 before anything is scraped rather than an empty list afterwards. What
    remains here is genuinely per-page: a field that matches nothing falls back
    to its ``default``, and a page that yields more rows than the cap is
    reported as ``truncated`` instead of silently short.

    Args:
        html: The page HTML, as produced by any backend.
        schema: The validated extraction schema from the request.

    Returns:
        An ``ExtractionResult``: one dict per matched base element, capped at
        ``MAX_EXTRACTED_ITEMS``, and whether that cap (or a per-field one) bit.
        Elements that yield no fields at all are skipped.
    """
    if not html:
        return ExtractionResult([])

    try:
        soup = BeautifulSoup(html, "lxml")
        base_elements = soup.select(schema.base_selector)
    except Exception as e:
        # Selectors are compiled at request time, so reaching here means the
        # page/parser broke, not the caller. Surface it as truncated-with-no-rows
        # rather than an innocent empty result.
        logger.warning("extraction_schema base selector %r failed: %s", schema.base_selector, e)
        return ExtractionResult([], truncated=True)

    state = _Budget()
    results: list[dict[str, Any]] = []
    for element in base_elements[:MAX_EXTRACTED_ITEMS]:
        item = _extract_item(element, schema.fields, state)
        if item:
            results.append(item)
    return ExtractionResult(
        results,
        truncated=state.truncated or len(base_elements) > MAX_EXTRACTED_ITEMS,
        field_errors=sorted(state.field_errors),
    )


class _Budget:
    """Tracks whether any cap bit, and which fields hit a genuine runtime error."""

    __slots__ = ("field_errors", "truncated")

    def __init__(self) -> None:
        self.truncated = False
        self.field_errors: set[str] = set()


def _extract_item(element: Tag, fields: list[ExtractionField], state: _Budget) -> dict[str, Any]:
    """Extract every field of one record from its base element."""
    item: dict[str, Any] = {}
    for field in fields:
        value = _extract_field(element, field, state)
        if value is not None:
            item[field.name] = value
    return item


def _extract_field(element: Tag, field: ExtractionField, state: _Budget) -> Any:
    """Extract one field, isolating failures to that field.

    A selector that correctly matches nothing is NOT an error — it never
    reaches this ``except``, it flows through the normal ``default``/empty-list
    path in ``_select``/``_extract_scalar``. Only a genuine failure while
    *running* an already-validated selector (or transforming its result) lands
    here, and that must be visible: at WARNING (never DEBUG, which this estate
    runs below and therefore never records), and recorded on ``state`` so the
    caller can see it in the response instead of a silently-defaulted field.
    """
    try:
        if field.type == "nested":
            nested = _select(element, field.selector, field.name, state)
            return _extract_item(nested[0], field.fields, state) if nested else field.default

        if field.type in ("list", "nested_list"):
            # Both fan out over every match; `list` treats each match as a flat
            # record and `nested_list` recurses, which is the same call here
            # because `_extract_item` already handles either field shape.
            matches = _select(element, field.selector, field.name, state)
            if len(matches) > MAX_EXTRACTED_LIST_ITEMS:
                state.truncated = True
                matches = matches[:MAX_EXTRACTED_LIST_ITEMS]
            return [_extract_item(el, field.fields, state) for el in matches]

        return _extract_scalar(element, field, state)
    except Exception as e:
        logger.warning("extraction_schema field %r failed to extract: %s", field.name, e)
        state.field_errors.add(field.name)
        # A `list` field must stay a list even on failure: handing back the
        # scalar `default` changed the declared type of that record key and
        # broke callers iterating it.
        return [] if field.type in ("list", "nested_list") else field.default


def _extract_scalar(element: Tag, field: ExtractionField, state: _Budget) -> Any:
    """Extract a text/attribute/html value from the field's target element."""
    if field.selector:
        selected = _select(element, field.selector, field.name, state)
        if not selected:
            return field.default
        target = selected[0]
    else:
        target = element

    value: str | None
    if field.type == "attribute":
        raw = target.get(field.attribute or "")
        # bs4 returns multi-valued attributes (class, rel) as a list.
        value = " ".join(raw) if isinstance(raw, list) else raw
    elif field.type == "html":
        value = str(target)
    else:  # "text"
        value = target.get_text(strip=True)

    if value is None:
        return field.default

    if field.transform == "lowercase":
        value = value.lower()
    elif field.transform == "uppercase":
        value = value.upper()
    elif field.transform == "strip":
        value = value.strip()

    if len(value) > MAX_EXTRACTED_VALUE_LENGTH:
        value = value[:MAX_EXTRACTED_VALUE_LENGTH]

    # An empty string is a real "matched but blank" answer, distinct from
    # "no match" — only the latter falls back to the default.
    return value


def _select(element: Tag, selector: str | None, field_name: str, state: _Budget) -> list[Tag]:
    """Run a CSS selector, treating a genuinely absent match as no match.

    ``selector`` was already accepted by ``soupsieve.compile`` at request
    validation time, so an exception here is NOT a bad selector — it is a
    runtime failure to actually run a valid one against this page's DOM. That
    is case (c) from the caller's point of view, and must not be
    indistinguishable from case (b) ("matched nothing", the empty ``[]`` this
    function returns in both branches) — hence the WARNING log and the record
    on ``state`` even though the return value is the same ``[]`` either way.
    """
    if not selector:
        return []
    try:
        return element.select(selector)
    except Exception as e:
        logger.warning(
            "extraction_schema field %r selector %r raised while running "
            "against the page (selector was valid CSS): %s",
            field_name,
            selector,
            e,
        )
        state.field_errors.add(field_name)
        return []


__all__ = [
    "MAX_EXTRACTED_ITEMS",
    "MAX_EXTRACTED_LIST_ITEMS",
    "MAX_EXTRACTED_VALUE_LENGTH",
    "ExtractionResult",
    "extract_structured",
]
