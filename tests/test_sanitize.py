"""Tests for control-char sanitization of scraped strings.

A page whose anchor text / image alt / title contains a C0 control char
(form-feed, vertical-tab, ...) previously raised a Pydantic ValidationError when
the router built the SAFE_STRING_PATTERN-constrained response → HTTP 500 (and for
/bulk, the whole batch). Scraped strings are now sanitized at the router boundary.
"""

import re

from custom_types.constraint_types import SAFE_STRING_PATTERN, sanitize_safe_string
from routers.scrape import _clean, _clean_opt, _convert_images, _convert_links
from schemas.scrape import ScrapeResponse, ScrapeStatus

_PATTERN = re.compile(SAFE_STRING_PATTERN)


class TestSanitizeHelper:
    def test_strips_c0_controls_and_del(self):
        assert sanitize_safe_string("Read\x0cmore\x1f\x08\x7f") == "Readmore"

    def test_keeps_tab_newline_cr(self):
        assert sanitize_safe_string("a\tb\nc\rd") == "a\tb\nc\rd"

    def test_non_str_passthrough(self):
        assert sanitize_safe_string(None) is None  # type: ignore[arg-type]

    def test_result_always_matches_pattern(self):
        for raw in ("\x00\x01\x02hi", "x\x0by\x0cz", "plain", "emoji 🎉 ok"):
            assert _PATTERN.match(sanitize_safe_string(raw)) is not None


class TestRouterBoundary:
    def test_convert_links_sanitizes(self):
        links = _convert_links(
            [
                {
                    "text": "Read\x0cmore",
                    "url": "https://x.com/a\x1f",
                    "rel": "nofollow",
                    "type": "external",
                }
            ]
        )
        assert links[0].text == "Readmore"
        assert links[0].url == "https://x.com/a"

    def test_convert_images_sanitizes(self):
        imgs = _convert_images(
            [{"src": "https://x.com/i.png", "alt": "cap\x0btion", "title": "t\x08"}]
        )
        assert imgs[0].alt == "caption"
        assert imgs[0].title == "t"

    def test_response_fields_sanitized_via_helpers(self):
        # Mirrors the router: scraped fields pass through _clean/_clean_opt first,
        # so a control char in title/scraper_used no longer raises.
        resp = ScrapeResponse(
            url=_clean("https://x.com"),
            title=_clean_opt("Hi\x0cthere"),
            status=ScrapeStatus.COMPLETED,
            scraper_used=_clean("httpx\x7f"),
            processing_time_seconds=0.1,
        )
        assert resp.title == "Hithere"
        assert resp.scraper_used == "httpx"
