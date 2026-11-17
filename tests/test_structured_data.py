"""Tests for embedded structured-data extraction (services.structured_data)."""

import services.structured_data as sd
from services.structured_data import extract_structured_data

# An article page carrying schema.org JSON-LD + OpenGraph + a Twitter card meta.
ARTICLE_HTML = """
<html>
<head>
  <title>Payments deep-dive</title>
  <script type="application/ld+json">
  {
    "@context": "https://schema.org",
    "@type": "Article",
    "headline": "How modern card networks settle payments",
    "author": {"@type": "Person", "name": "Jane Doe"},
    "datePublished": "2026-01-02"
  }
  </script>
  <meta property="og:title" content="How card networks settle" />
  <meta property="og:type" content="article" />
</head>
<body><article><p>Body.</p></article></body>
</html>
"""

# A product page using JSON-LD Product with an offer.
PRODUCT_HTML = """
<html><head>
  <script type="application/ld+json">
  {"@context":"https://schema.org","@type":"Product","name":"Widget",
   "offers":{"@type":"Offer","price":"19.99","priceCurrency":"USD"}}
  </script>
</head><body>Widget page</body></html>
"""


def test_empty_input_returns_empty_dict():
    assert extract_structured_data("") == {}
    assert extract_structured_data("   ") == {}


def test_jsonld_article_author_and_date_parsed():
    data = extract_structured_data(ARTICLE_HTML, "https://example.com/article")
    assert "json-ld" in data
    article = data["json-ld"][0]
    assert article["@type"] == "Article"
    assert article["author"]["name"] == "Jane Doe"
    assert article["datePublished"] == "2026-01-02"


def test_opengraph_parsed():
    data = extract_structured_data(ARTICLE_HTML, "https://example.com/article")
    assert "opengraph" in data
    og = data["opengraph"][0]
    assert og.get("og:title") == "How card networks settle"


def test_jsonld_product_offer_parsed():
    data = extract_structured_data(PRODUCT_HTML, "https://shop.example.com/widget")
    product = data["json-ld"][0]
    assert product["@type"] == "Product"
    assert product["offers"]["price"] == "19.99"


def test_empty_syntaxes_are_dropped():
    # A page with no embedded structured data at all → empty dict, not a dict of
    # empty-list syntaxes.
    data = extract_structured_data("<html><body><p>plain</p></body></html>", "https://x.test")
    assert data == {}


def test_malformed_jsonld_does_not_raise():
    html = (
        '<html><head><script type="application/ld+json">{ not valid json,, }</script>'
        "</head><body>x</body></html>"
    )
    # errors="log" inside extruct skips the bad block; never raises.
    data = extract_structured_data(html, "https://x.test")
    assert isinstance(data, dict)


def test_extruct_exception_returns_empty(monkeypatch):
    """An unexpected extruct failure must never break a scrape."""

    def _boom(*_a, **_k):
        raise RuntimeError("extruct exploded")

    monkeypatch.setattr(sd.extruct, "extract", _boom)
    assert extract_structured_data(ARTICLE_HTML, "https://x.test") == {}


def test_oversized_payload_trimmed_to_priority(monkeypatch):
    """When the full result blows the byte budget, only json-ld+opengraph survive."""
    big = {
        "json-ld": [{"@type": "Article", "headline": "small"}],
        "opengraph": [{"og:title": "small"}],
        "microdata": [{"blob": "x" * 10}],
        "rdfa": [{"blob": "y" * 10}],
    }
    monkeypatch.setattr(sd.extruct, "extract", lambda *a, **k: big)
    monkeypatch.setattr(sd, "_MAX_BYTES", 120)  # forces the trim path
    data = extract_structured_data(ARTICLE_HTML, "https://x.test")
    assert set(data.keys()) <= {"json-ld", "opengraph"}
    assert "json-ld" in data


def test_oversized_even_after_trim_dropped(monkeypatch):
    """If even json-ld+opengraph exceed the budget, structured_data is dropped."""
    big = {"json-ld": [{"blob": "x" * 1000}], "opengraph": [{"og": "y" * 1000}]}
    monkeypatch.setattr(sd.extruct, "extract", lambda *a, **k: big)
    monkeypatch.setattr(sd, "_MAX_BYTES", 50)
    assert extract_structured_data(ARTICLE_HTML, "https://x.test") == {}
