"""Tests for declarative CSS extraction (services.extraction_schema)."""

import logging

import pytest
from pydantic import ValidationError

from schemas.scrape import (
    MAX_EXTRACTION_DEPTH,
    MAX_EXTRACTION_FIELDS,
    ExtractionField,
    ExtractionSchema,
    ScrapeRequest,
)
from services.extraction_schema import (
    MAX_EXTRACTED_ITEMS,
    MAX_EXTRACTED_LIST_ITEMS,
    MAX_EXTRACTED_VALUE_LENGTH,
    extract_structured,
)

LISTING_HTML = """
<!DOCTYPE html><html><body>
  <div class="results">
    <article class="card" data-id="1">
      <h2 class="name">Acme Payments</h2>
      <span class="city">Berlin</span>
      <a class="site" href="https://acme.example/pricing" rel="noopener nofollow">Visit</a>
      <ul class="tags"><li>fintech</li><li>b2b</li></ul>
    </article>
    <article class="card" data-id="2">
      <h2 class="name">Globex Settlement</h2>
      <span class="city">Dublin</span>
      <a class="site" href="https://globex.example">Visit</a>
      <ul class="tags"><li>payments</li></ul>
    </article>
    <article class="card" data-id="3">
      <h2 class="name">Initech</h2>
      <!-- no city, no link: exercises defaults -->
    </article>
  </div>
</body></html>
"""


def _listing_schema(**overrides) -> ExtractionSchema:
    payload = {
        "name": "vendors",
        "baseSelector": "article.card",
        "fields": [
            {"name": "company", "selector": "h2.name", "type": "text"},
            {"name": "city", "selector": "span.city", "type": "text", "default": "unknown"},
            {
                "name": "website",
                "selector": "a.site",
                "type": "attribute",
                "attribute": "href",
            },
        ],
    }
    payload.update(overrides)
    return ExtractionSchema.model_validate(payload)


def _rows(html: str, schema: ExtractionSchema) -> list[dict]:
    """Just the records; the truncation flag is asserted where it matters."""
    return extract_structured(html, schema).items


class TestExtraction:
    def test_extracts_one_record_per_base_element(self):
        rows = _rows(LISTING_HTML, _listing_schema())
        assert len(rows) == 3
        assert rows[0]["company"] == "Acme Payments"
        assert rows[1]["company"] == "Globex Settlement"

    def test_attribute_extraction(self):
        rows = _rows(LISTING_HTML, _listing_schema())
        assert rows[0]["website"] == "https://acme.example/pricing"

    def test_default_applies_when_selector_matches_nothing(self):
        rows = _rows(LISTING_HTML, _listing_schema())
        assert rows[2]["city"] == "unknown"

    def test_field_with_no_match_and_no_default_is_omitted(self):
        rows = _rows(LISTING_HTML, _listing_schema())
        assert "website" not in rows[2]

    def test_multivalued_attribute_is_joined_not_stringified_as_a_list(self):
        schema = ExtractionSchema.model_validate(
            {
                "baseSelector": "article.card",
                "fields": [
                    {"name": "rel", "selector": "a.site", "type": "attribute", "attribute": "rel"}
                ],
            }
        )
        rows = _rows(LISTING_HTML, schema)
        assert rows[0]["rel"] == "noopener nofollow"

    def test_html_type_returns_markup(self):
        schema = ExtractionSchema.model_validate(
            {
                "baseSelector": "article.card",
                "fields": [{"name": "raw", "selector": "h2.name", "type": "html"}],
            }
        )
        assert _rows(LISTING_HTML, schema)[0]["raw"].startswith('<h2 class="name"')

    def test_transform_is_applied(self):
        schema = _listing_schema(
            fields=[
                {"name": "company", "selector": "h2.name", "type": "text", "transform": "uppercase"}
            ]
        )
        assert _rows(LISTING_HTML, schema)[0]["company"] == "ACME PAYMENTS"

    def test_field_without_selector_reads_the_base_element(self):
        schema = ExtractionSchema.model_validate(
            {
                "baseSelector": "article.card",
                "fields": [{"name": "id", "type": "attribute", "attribute": "data-id"}],
            }
        )
        assert [r["id"] for r in _rows(LISTING_HTML, schema)] == ["1", "2", "3"]

    def test_nested_list_of_children(self):
        schema = ExtractionSchema.model_validate(
            {
                "baseSelector": "article.card",
                "fields": [
                    {"name": "company", "selector": "h2.name", "type": "text"},
                    {
                        "name": "tags",
                        "selector": "ul.tags li",
                        "type": "list",
                        "fields": [{"name": "label", "type": "text"}],
                    },
                ],
            }
        )
        rows = _rows(LISTING_HTML, schema)
        assert rows[0]["tags"] == [{"label": "fintech"}, {"label": "b2b"}]
        assert rows[2]["tags"] == []

    def test_nested_object(self):
        schema = ExtractionSchema.model_validate(
            {
                "baseSelector": "article.card",
                "fields": [
                    {
                        "name": "link",
                        "selector": "a.site",
                        "type": "nested",
                        "fields": [
                            {"name": "label", "type": "text"},
                            {"name": "href", "type": "attribute", "attribute": "href"},
                        ],
                    }
                ],
            }
        )
        rows = _rows(LISTING_HTML, schema)
        assert rows[0]["link"] == {"label": "Visit", "href": "https://acme.example/pricing"}


class TestFailureIsolation:
    """A bad schema degrades to empty output — it never 500s the whole scrape."""

    def test_base_selector_matching_nothing_returns_empty(self):
        schema = ExtractionSchema.model_validate(
            {"baseSelector": "table.nope", "fields": [{"name": "x", "type": "text"}]}
        )
        assert _rows(LISTING_HTML, schema) == []

    def test_invalid_css_in_base_selector_is_rejected_at_validation(self):
        """Caller error must fail loudly, BEFORE a scrape is paid for.

        This used to be swallowed into ``extracted_data: []`` after a full
        scrape — indistinguishable from "the page had no matching rows", so a
        typo in a selector looked like an empty directory.
        """
        with pytest.raises(ValidationError, match="not a valid CSS selector"):
            ExtractionSchema.model_validate(
                {"baseSelector": "div:::broken", "fields": [{"name": "x", "type": "text"}]}
            )

    def test_invalid_css_in_field_selector_is_rejected_at_validation(self):
        with pytest.raises(ValidationError, match="not a valid CSS selector"):
            _listing_schema(
                fields=[
                    {"name": "company", "selector": "h2.name", "type": "text"},
                    {"name": "broken", "selector": "span:::bad", "type": "text"},
                ]
            )

    def test_invalid_css_nested_deep_in_the_schema_is_still_rejected(self):
        """The walk must cover child fields, not just the top level."""
        with pytest.raises(ValidationError, match="not a valid CSS selector"):
            _listing_schema(
                fields=[
                    {
                        "name": "tags",
                        "selector": "ul.tags",
                        "type": "list",
                        "fields": [{"name": "label", "selector": "li:::bad", "type": "text"}],
                    }
                ]
            )

    def test_empty_html_returns_empty(self):
        assert _rows("", _listing_schema()) == []


class TestRuntimeFieldFailuresAreVisibleNotDebugSwallowed:
    """A field whose (already CSS-valid) selector raises while actually
    running against the page — this estate's LOG_LEVEL is INFO, so a DEBUG log
    for that is no record at all. It must be distinguishable from a selector
    that correctly matched nothing (still a normal default/empty result).
    """

    def test_selector_execution_failure_is_reported_via_select(self, monkeypatch, caplog):
        """A valid selector that raises inside element.select() itself."""
        from bs4.element import Tag

        original_select = Tag.select

        def flaky_select(self, selector, *args, **kwargs):
            if selector == "span.city":
                raise RuntimeError("simulated soupsieve runtime failure")
            return original_select(self, selector, *args, **kwargs)

        monkeypatch.setattr(Tag, "select", flaky_select)

        with caplog.at_level(logging.WARNING, logger="services.extraction_schema"):
            result = extract_structured(LISTING_HTML, _listing_schema())

        # Extraction still degrades gracefully -- it never 500s the scrape.
        assert result.items[0]["city"] == "unknown"
        # But this is NOT "the page has no city" -- it must be recorded and
        # logged loudly, not silently defaulted with zero trace.
        assert "city" in result.field_errors
        assert any(
            rec.levelno >= logging.WARNING and "city" in rec.getMessage() for rec in caplog.records
        )

    def test_extraction_failure_outside_select_is_reported(self, monkeypatch, caplog):
        """A valid selector matches fine, but turning the match into a value
        raises (e.g. get_text() blowing up) -- a different code path than the
        selector-execution failure above, and must be caught the same way.
        """
        from bs4.element import Tag

        original_get_text = Tag.get_text

        def flaky_get_text(self, *args, **kwargs):
            if self.name == "h2":
                raise RuntimeError("simulated get_text failure")
            return original_get_text(self, *args, **kwargs)

        monkeypatch.setattr(Tag, "get_text", flaky_get_text)

        with caplog.at_level(logging.WARNING, logger="services.extraction_schema"):
            result = extract_structured(LISTING_HTML, _listing_schema())

        # "company" has no default -> a genuinely failed field is omitted,
        # same as "matched nothing with no default" -- but unlike that case it
        # must show up in field_errors and in a WARNING-level log line.
        assert "company" not in result.items[0]
        assert "company" in result.field_errors
        assert any(
            rec.levelno >= logging.WARNING and "company" in rec.getMessage()
            for rec in caplog.records
        )

    def test_a_selector_that_correctly_matches_nothing_is_not_a_field_error(self):
        """Case (b) must stay a normal empty result -- never in field_errors."""
        schema = ExtractionSchema.model_validate(
            {"baseSelector": "table.nope", "fields": [{"name": "x", "type": "text"}]}
        )
        result = extract_structured(LISTING_HTML, schema)
        assert result.field_errors == []


class TestBounds:
    def test_item_count_is_capped(self):
        cards = "".join(
            f'<article class="card"><h2 class="name">C{i}</h2></article>'
            for i in range(MAX_EXTRACTED_ITEMS + 25)
        )
        rows = _rows(f"<html><body>{cards}</body></html>", _listing_schema())
        assert len(rows) == MAX_EXTRACTED_ITEMS

    def test_hitting_the_item_cap_is_reported_not_silent(self):
        """1,000 of 5,000 rows handed back as a complete answer is a wrong answer."""
        cards = "".join(
            f'<article class="card"><h2 class="name">C{i}</h2></article>'
            for i in range(MAX_EXTRACTED_ITEMS + 25)
        )
        result = extract_structured(f"<html><body>{cards}</body></html>", _listing_schema())
        assert len(result.items) == MAX_EXTRACTED_ITEMS
        assert result.truncated is True

    def test_an_uncapped_page_is_not_flagged_as_truncated(self):
        assert extract_structured(LISTING_HTML, _listing_schema()).truncated is False

    def test_list_field_fanout_is_bounded_and_flagged(self):
        """MAX_EXTRACTED_ITEMS bounds RECORDS; a list field fans out per record.

        Without a per-field cap one `list` selector matching every node made
        each record unbounded, so the response was too — and it arrived as a
        clean 200.
        """
        tags = "".join(f"<li>t{i}</li>" for i in range(MAX_EXTRACTED_LIST_ITEMS + 50))
        html = f'<article class="card"><ul class="tags">{tags}</ul></article>'
        schema = _listing_schema(
            fields=[
                {
                    "name": "tags",
                    "selector": "ul.tags li",
                    "type": "list",
                    "fields": [{"name": "label", "type": "text"}],
                }
            ]
        )
        result = extract_structured(html, schema)
        assert len(result.items[0]["tags"]) == MAX_EXTRACTED_LIST_ITEMS
        assert result.truncated is True

    def test_value_length_is_capped(self):
        blob = "x" * (MAX_EXTRACTED_VALUE_LENGTH + 500)
        html = f'<article class="card"><h2 class="name">{blob}</h2></article>'
        rows = _rows(html, _listing_schema())
        assert len(rows[0]["company"]) == MAX_EXTRACTED_VALUE_LENGTH


class TestSchemaValidation:
    def test_crawl4ai_camelcase_dialect_is_accepted(self):
        """An existing crawl4ai schema must paste in verbatim."""
        schema = ExtractionSchema.model_validate(
            {"baseSelector": "article.card", "fields": [{"name": "x", "type": "text"}]}
        )
        assert schema.base_selector == "article.card"

    def test_snake_case_is_also_accepted(self):
        schema = ExtractionSchema.model_validate(
            {"base_selector": "article.card", "fields": [{"name": "x", "type": "text"}]}
        )
        assert schema.base_selector == "article.card"

    def test_attribute_type_requires_an_attribute_name(self):
        with pytest.raises(ValidationError, match="requires 'attribute'"):
            ExtractionField.model_validate({"name": "x", "type": "attribute"})

    def test_container_type_requires_child_fields(self):
        with pytest.raises(ValidationError, match="requires 'fields'"):
            ExtractionField.model_validate({"name": "x", "type": "list", "selector": "li"})

    def test_container_type_requires_a_selector(self):
        with pytest.raises(ValidationError, match="requires 'selector'"):
            ExtractionField.model_validate(
                {"name": "x", "type": "list", "fields": [{"name": "y", "type": "text"}]}
            )

    def test_scalar_type_must_not_declare_child_fields(self):
        with pytest.raises(ValidationError, match="must not declare 'fields'"):
            ExtractionField.model_validate(
                {"name": "x", "type": "text", "fields": [{"name": "y", "type": "text"}]}
            )

    def test_schema_requires_at_least_one_field(self):
        with pytest.raises(ValidationError):
            ExtractionSchema.model_validate({"baseSelector": "div", "fields": []})

    def test_total_field_count_is_bounded(self):
        with pytest.raises(ValidationError, match="limit is"):
            ExtractionSchema.model_validate(
                {
                    "baseSelector": "div",
                    "fields": [
                        {"name": f"f{i}", "type": "text"} for i in range(MAX_EXTRACTION_FIELDS + 1)
                    ],
                }
            )

    def test_nesting_depth_is_bounded(self):
        field = {"name": "leaf", "type": "text"}
        for _ in range(MAX_EXTRACTION_DEPTH):
            field = {"name": "n", "type": "nested", "selector": "div", "fields": [field]}
        with pytest.raises(ValidationError, match="limit is"):
            ExtractionSchema.model_validate({"baseSelector": "div", "fields": [field]})

    def test_computed_and_regex_types_are_rejected(self):
        """Both upstream types are deliberately unsupported; they must 422, not
        silently no-op, or a caller ports a crawl4ai schema and gets quiet gaps."""
        for unsupported in ("computed", "regex"):
            with pytest.raises(ValidationError):
                ExtractionField.model_validate({"name": "x", "type": unsupported})

    def test_request_accepts_a_schema(self):
        request = ScrapeRequest.model_validate(
            {
                "url": "https://example.com",
                "extraction_schema": {
                    "baseSelector": "article.card",
                    "fields": [{"name": "x", "type": "text"}],
                },
            }
        )
        assert request.extraction_schema is not None
        assert request.extraction_schema.base_selector == "article.card"

    def test_realistic_css_selectors_survive_validation_and_run(self):
        """Combinators, attribute selectors, pseudo-classes and quotes must pass.

        SAFE_STRING_PATTERN only bars control characters, but the feature is
        worthless if a real-world selector 422s, so pin it.
        """
        html = """
        <div class="results">
          <div class="item"><h3>One</h3><a href="/p/1" data-kind="primary">go</a></div>
          <div class="item"><h3>Two</h3><a href="/other/2" data-kind="secondary">go</a></div>
        </div>
        """
        schema = ExtractionSchema.model_validate(
            {
                "baseSelector": "div.results > div.item",
                "fields": [
                    {"name": "title", "selector": "h3:nth-of-type(1)", "type": "text"},
                    {
                        "name": "href",
                        "selector": 'a[href^="/p/"][data-kind="primary"]',
                        "type": "attribute",
                        "attribute": "href",
                    },
                ],
            }
        )
        rows = _rows(html, schema)
        assert [r["title"] for r in rows] == ["One", "Two"]
        assert rows[0]["href"] == "/p/1"
        assert "href" not in rows[1]
