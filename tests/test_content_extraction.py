"""Tests for main-content extraction (services.content_extraction)."""

import importlib

import services.content_extraction as ce
from services.content_extraction import (
    _boilerplate_stripped_text,
    extract_main_text,
    extract_markdown,
)

# A real-ish article buried in site chrome. The body is long enough that the
# primary trafilatura path keeps it (and passes the recall guard).
ARTICLE_HTML = """
<!DOCTYPE html>
<html>
<head><title>Payments deep-dive</title></head>
<body>
  <header><a href="/">BrandLogo</a></header>
  <nav>Home Products Pricing About Contact Login</nav>
  <div class="cookie-bar">We use cookies. Accept all cookies to continue.</div>
  <main>
    <article>
      <h1>How modern card networks settle payments</h1>
      <p>Card settlement is the multi-step process by which a merchant ultimately
         receives funds for a transaction. After authorization, the acquiring bank
         batches transactions and submits them to the card network for clearing.</p>
      <p>During clearing the network calculates interchange and forwards the
         transaction to the issuing bank, which posts it to the cardholder account.
         Funds then move during settlement, typically the next business day.</p>
    </article>
  </main>
  <footer>Copyright 2026 BrandLogo Inc. Privacy Terms Sitemap</footer>
</body>
</html>
"""

# A directory/listing page whose substance is a table of companies.
DIRECTORY_HTML = """
<html>
<body>
  <nav>Skip to content Main menu Search Donate Create account Log in</nav>
  <h1>Largest software companies</h1>
  <table>
    <tr><th>Company</th><th>Revenue</th><th>Country</th></tr>
    <tr><td>Acme Corp</td><td>100</td><td>USA</td></tr>
    <tr><td>Globex</td><td>80</td><td>Germany</td></tr>
  </table>
  <footer>Retrieved from the free encyclopedia</footer>
</body>
</html>
"""

# A non-article layout (status-style page) where trafilatura typically yields
# nothing, so the boilerplate-stripped baseline must carry it.
NON_ARTICLE_HTML = """
<html>
<body>
  <header>StatusPage</header>
  <div>Status: operational</div>
  <div>Region: eu-west</div>
  <div>Last incident: none</div>
  <footer>powered by status</footer>
</body>
</html>
"""

FORM_ONLY_HTML = """
<html><body>
  <nav>Login</nav>
  <form>
    <input name="email" />
    <input name="password" type="password" />
    <button type="submit">Sign in</button>
  </form>
</body></html>
"""


def test_empty_input_returns_empty():
    assert extract_main_text("") == ""
    assert extract_main_text("   ") != "   "  # whitespace-only HTML yields no text


def test_article_drops_chrome_keeps_body():
    text = extract_main_text(ARTICLE_HTML, "https://example.com/article")
    # Real body survives.
    assert "Card settlement is the multi-step process" in text
    assert "clearing the network calculates interchange" in text
    # Site chrome is gone.
    assert "Main menu" not in text
    assert "Accept all cookies" not in text
    assert "Privacy Terms Sitemap" not in text
    assert "BrandLogo" not in text


def test_directory_keeps_table_content_drops_chrome():
    text = extract_main_text(DIRECTORY_HTML, "https://example.com/list")
    # The company rows (the whole point of the page) survive.
    assert "Acme Corp" in text
    assert "Globex" in text
    assert "Germany" in text
    # Nav chrome is stripped.
    assert "Create account" not in text
    assert "Skip to content" not in text


def test_non_article_falls_back_to_baseline():
    # trafilatura returns ~nothing here; the baseline must still surface the text.
    text = extract_main_text(NON_ARTICLE_HTML, "https://status.example.com")
    assert "Status: operational" in text
    assert "Region: eu-west" in text
    assert "powered by status" not in text  # footer stripped by the baseline


def test_form_only_page_does_not_crash():
    text = extract_main_text(FORM_ONLY_HTML, "https://example.com/login")
    # form + nav are boilerplate; result is empty but extraction must not raise.
    assert "Sign in" not in text
    assert isinstance(text, str)


def test_baseline_preserves_table_row_grouping():
    # Directly exercise the baseline: each row's cells stay together on one line
    # joined with " | " (instead of one cell per line).
    text = _boilerplate_stripped_text(DIRECTORY_HTML)
    assert "Acme Corp | 100 | USA" in text
    assert "Globex | 80 | Germany" in text


def test_baseline_strips_structural_chrome():
    text = _boilerplate_stripped_text(ARTICLE_HTML)
    assert "How modern card networks settle payments" in text
    assert "Main menu" not in text  # nav removed
    assert "Copyright 2026" not in text  # footer removed
    assert "BrandLogo" not in text  # header removed


def test_trafilatura_exception_falls_back_to_baseline(monkeypatch):
    """If trafilatura raises, the scrape must not break — the baseline carries it."""

    def _boom(*_args, **_kwargs):
        raise RuntimeError("trafilatura blew up")

    monkeypatch.setattr(ce.trafilatura, "extract", _boom)
    text = extract_main_text(ARTICLE_HTML, "https://example.com/article")
    assert "Card settlement is the multi-step process" in text
    assert "Main menu" not in text  # baseline still strips chrome


def test_thin_trafilatura_result_below_ratio_uses_baseline(monkeypatch):
    """A trafilatura result that is *above* MIN_ABS_CHARS but retains too little of
    the baseline (e.g. a directory page the article extractor over-pruned) is
    rejected by the recall-ratio guard in favor of the baseline."""
    # Large directory page → big baseline; trafilatura returns a 300-char sliver
    # that clears MIN_ABS_CHARS(200) yet is < MIN_RATIO(0.10) of the baseline.
    rows = "".join(
        f"<tr><td>Company {i}</td><td>{i * 10}</td><td>Country {i}</td></tr>" for i in range(200)
    )
    big_html = f"<html><body><nav>menu</nav><table>{rows}</table></body></html>"
    sliver = "A" * 300
    monkeypatch.setattr(ce.trafilatura, "extract", lambda *a, **k: sliver)
    text = extract_main_text(big_html, "https://example.com/list")
    assert "Company 0" in text  # baseline returned, not the 300-char sliver
    assert sliver not in text


def test_thin_result_below_min_abs_chars_uses_baseline(monkeypatch):
    """Even above the ratio, a result under MIN_ABS_CHARS is too thin to trust."""
    monkeypatch.setattr(ce.trafilatura, "extract", lambda *a, **k: "tiny")
    text = extract_main_text(NON_ARTICLE_HTML, "https://status.example.com")
    assert "Status: operational" in text


def test_role_navigation_is_stripped():
    html = (
        '<html><body><div role="navigation">Jump menu junk links</div>'
        "<article><p>The genuine article body lives here and is long enough.</p>"
        "</article></body></html>"
    )
    text = _boilerplate_stripped_text(html)
    assert "genuine article body" in text
    assert "Jump menu junk" not in text


def test_baseline_strips_cookie_consent_banners():
    """Cookie/consent/GDPR banners and modal dialogs are dropped on the baseline
    path — the long-tail of non-article pages where trafilatura yields nothing and
    the boilerplate-stripped baseline carries the page (so the banner would leak)."""
    html = (
        "<html><body>"
        '<div class="cookie-bar">We use cookies. Accept all cookies to continue.</div>'
        '<div id="gdpr-consent">Manage your GDPR preferences here.</div>'
        '<div role="dialog">Subscribe to our newsletter popup.</div>'
        '<div aria-modal="true">Modal overlay junk text.</div>'
        "<article><p>The genuine article body lives here and is plenty long.</p>"
        "</article></body></html>"
    )
    text = _boilerplate_stripped_text(html)
    assert "genuine article body" in text
    assert "Accept all cookies" not in text
    assert "GDPR preferences" not in text
    assert "newsletter popup" not in text
    assert "Modal overlay junk" not in text


def test_baseline_keeps_body_text_mentioning_cookies():
    """Over-strip guard: only chrome *containers* (cookie/consent/gdpr class/id,
    dialog/modal role) are dropped — plain body copy that merely mentions cookies
    is preserved, because we match on the container attribute, not on text."""
    html = (
        "<html><body><article>"
        "<p>This shortbread recipe calls for two cups of flour and a jar of cookies.</p>"
        "</article></body></html>"
    )
    text = _boilerplate_stripped_text(html)
    assert "jar of cookies" in text


def test_consent_banner_stripped_on_forced_baseline_path(monkeypatch):
    """Through the public entrypoint: when trafilatura yields too little and the
    baseline carries the page, consent banners are still removed from text_content."""
    monkeypatch.setattr(ce.trafilatura, "extract", lambda *a, **k: "")
    html = (
        "<html><body>"
        '<div class="cookie-consent-banner">Accept all cookies to continue.</div>'
        "<div>Service status: all systems operational across every region today.</div>"
        "</body></html>"
    )
    text = extract_main_text(html, "https://status.example.com")
    assert "all systems operational" in text
    assert "Accept all cookies" not in text


def test_nested_table_rows_linearized():
    """Innermost-first row replacement folds a nested table's rows into the outer
    cell without losing either row's grouping."""
    html = """
    <html><body><table>
      <tr><td>Outer A</td><td>
        <table><tr><td>Inner 1</td><td>Inner 2</td></tr></table>
      </td></tr>
    </table></body></html>
    """
    text = _boilerplate_stripped_text(html)
    assert "Inner 1 | Inner 2" in text
    assert "Outer A" in text


def test_env_overrides_recall_guard(monkeypatch):
    """MIN_ABS_CHARS / MIN_RATIO are env-overridable for ops without a redeploy."""
    monkeypatch.setenv("WEBSCRAPER_EXTRACT_MIN_ABS_CHARS", "7")
    monkeypatch.setenv("WEBSCRAPER_EXTRACT_MIN_RATIO", "0.42")
    try:
        importlib.reload(ce)
        assert ce.MIN_ABS_CHARS == 7
        assert ce.MIN_RATIO == 0.42
    finally:
        monkeypatch.delenv("WEBSCRAPER_EXTRACT_MIN_ABS_CHARS", raising=False)
        monkeypatch.delenv("WEBSCRAPER_EXTRACT_MIN_RATIO", raising=False)
        importlib.reload(ce)


def test_bad_env_override_falls_back_to_default(monkeypatch):
    """A typo'd numeric override must NOT crash the service at import."""
    monkeypatch.setenv("WEBSCRAPER_EXTRACT_MIN_ABS_CHARS", "not-a-number")
    monkeypatch.setenv("WEBSCRAPER_EXTRACT_MIN_RATIO", "")
    try:
        importlib.reload(ce)  # would raise ValueError before the _num_env guard
        assert ce.MIN_ABS_CHARS == 200
        assert ce.MIN_RATIO == 0.10
    finally:
        monkeypatch.delenv("WEBSCRAPER_EXTRACT_MIN_ABS_CHARS", raising=False)
        monkeypatch.delenv("WEBSCRAPER_EXTRACT_MIN_RATIO", raising=False)
        importlib.reload(ce)


def test_baseline_empty_input_returns_empty():
    """The recall-guard safety net short-circuits on empty input."""
    assert _boilerplate_stripped_text("") == ""


def test_baseline_swallows_parser_error(monkeypatch):
    """If the lxml parser chokes, the safety net returns '' rather than raising."""

    def _boom(*_a, **_k):
        raise ValueError("lxml exploded")

    monkeypatch.setattr(ce, "BeautifulSoup", _boom)
    assert _boilerplate_stripped_text("<html><body>x</body></html>") == ""


def test_extract_main_text_survives_unparseable_baseline(monkeypatch):
    """If both trafilatura and the baseline fail, return '' — never raise."""

    def _boom(*_a, **_k):
        raise ValueError("lxml exploded")

    monkeypatch.setattr(ce.trafilatura, "extract", _boom)
    monkeypatch.setattr(ce, "BeautifulSoup", _boom)
    assert extract_main_text("<html><body>x</body></html>") == ""


# --- Markdown extraction (LLM/RAG-ready output) ----------------------------------


def test_markdown_empty_input_returns_none():
    assert extract_markdown("") is None
    assert extract_markdown("   ") is None  # whitespace-only HTML yields no content


def test_markdown_article_has_headings_and_body():
    md = extract_markdown(ARTICLE_HTML, "https://example.com/article")
    assert md is not None
    # Markdown heading marker present and body preserved.
    assert "#" in md
    assert "How modern card networks settle payments" in md
    assert "Card settlement is the multi-step process" in md
    # Site chrome stripped here too.
    assert "Main menu" not in md
    assert "Accept all cookies" not in md


def test_markdown_directory_keeps_table_pipes():
    md = extract_markdown(DIRECTORY_HTML, "https://example.com/list")
    assert md is not None
    # Table content survives (trafilatura renders rows as Markdown table pipes).
    assert "Acme Corp" in md
    assert "Globex" in md
    assert "|" in md


def test_markdown_backstop_used_when_trafilatura_thin(monkeypatch):
    """When trafilatura returns a sub-threshold sliver, the markdownify backstop
    over the chrome-stripped soup carries the page (and still drops chrome)."""
    monkeypatch.setattr(ce.trafilatura, "extract", lambda *a, **k: "tiny")
    md = extract_markdown(ARTICLE_HTML, "https://example.com/article")
    assert md is not None
    assert "How modern card networks settle payments" in md
    assert "Main menu" not in md  # backstop strips nav
    assert "Privacy Terms Sitemap" not in md  # backstop strips footer


def test_markdown_trafilatura_exception_falls_back(monkeypatch):
    """A trafilatura crash must not break extraction — the backstop carries it."""

    def _boom(*_a, **_k):
        raise RuntimeError("trafilatura blew up")

    monkeypatch.setattr(ce.trafilatura, "extract", _boom)
    md = extract_markdown(ARTICLE_HTML, "https://example.com/article")
    assert md is not None
    assert "How modern card networks settle payments" in md


def test_markdown_total_failure_returns_none(monkeypatch):
    """If both trafilatura and the soup parse fail, return None — never raise."""

    def _boom(*_a, **_k):
        raise ValueError("boom")

    monkeypatch.setattr(ce.trafilatura, "extract", _boom)
    monkeypatch.setattr(ce, "BeautifulSoup", _boom)
    assert extract_markdown("<html><body>x</body></html>") is None
