"""Main-content extraction from page HTML.

Turns a page's raw HTML into clean reading text — stripping site chrome (nav,
header, footer, cookie bars) while preserving the article body AND data
tables/directory listings. Shared by both the httpx and Playwright scrapers so
they emit identical ``text_content``.

Strategy (robust across articles, directories, tables, SPAs, login pages):

1. Primary — ``trafilatura.extract`` with recall favored and tables + links
   kept. ``include_tables=True`` is the specific flag that stops the
   article-tuned extractor from discarding the very tables a directory page is
   made of.
2. Backstop — a gentle "boilerplate-stripped" text pass that decomposes
   ``nav/header/footer/aside/form`` (plus script/style/noscript), drops
   cookie/consent/GDPR banners and modal dialogs by class/id/role/aria, and
   linearizes table rows as ``cell | cell | cell`` so row grouping survives.

A recall guard picks between them: trafilatura is trusted only when it returns a
meaningful share of the boilerplate-stripped baseline; otherwise the baseline is
returned. This guarantees we never regress to empty (the old ``networkidle``
failure) nor to chrome-dominated text.
"""

import logging
import os

import trafilatura
from bs4 import BeautifulSoup
from markdownify import markdownify as _html_to_markdown

logger = logging.getLogger(__name__)


def _num_env(name: str, default: str, cast):
    """Read a numeric tunable from the environment, tolerant of bad values.

    A typo'd override (e.g. ``MIN_ABS_CHARS=abc``) must not crash the service at
    import — fall back to the documented default and warn instead.
    """
    raw = os.getenv(name, default)
    try:
        return cast(raw)
    except TypeError, ValueError:
        logger.warning("Invalid %s=%r; using default %s", name, raw, default)
        return cast(default)


# Recall-guard tunables (env-overridable for ops without a redeploy).
# - MIN_ABS_CHARS: below this, a trafilatura result is too thin to trust on its own.
# - MIN_RATIO: trafilatura must retain at least this fraction of the gently-cleaned
#   baseline, else it likely over-pruned a non-article page (e.g. a directory table).
MIN_ABS_CHARS = _num_env("WEBSCRAPER_EXTRACT_MIN_ABS_CHARS", "200", int)
MIN_RATIO = _num_env("WEBSCRAPER_EXTRACT_MIN_RATIO", "0.10", float)

# Non-content structural elements removed before the baseline text pass.
_BOILERPLATE_TAGS = ("script", "style", "noscript", "nav", "header", "footer", "aside", "form")

# Cookie/consent/GDPR banners and modal dialogs are non-content chrome that
# trafilatura drops on real articles, but they survive the structural-tag pass
# above on non-article pages that fall back to this baseline. Match the common,
# low-false-positive container patterns by class/id substring (case-insensitive)
# + ARIA role/modal. Kept deliberately narrow (``cookie``/``consent``/``gdpr``
# are overwhelmingly banner containers, not body copy); the recall guard in
# ``extract_main_text`` is the backstop if a page ever over-prunes here.
_CONSENT_SELECTORS = (
    '[class*="cookie" i]',
    '[id*="cookie" i]',
    '[class*="consent" i]',
    '[id*="consent" i]',
    '[class*="gdpr" i]',
    '[id*="gdpr" i]',
    '[role="dialog"]',
    '[role="alertdialog"]',
    '[role="banner"]',
    '[aria-modal="true"]',
)


def extract_main_text(html: str, url: str | None = None) -> str:
    """Extract clean main-content text from a page's HTML.

    Args:
        html: Raw page HTML (for Playwright, the rendered ``page.content()``).
        url: Source URL, passed to trafilatura to improve link resolution.

    Returns:
        Clean text with site chrome removed. Empty string only for empty input.
    """
    if not html:
        return ""

    try:
        extracted = (
            trafilatura.extract(
                html,
                include_comments=False,
                include_tables=True,
                include_links=True,
                favor_recall=True,
                output_format="txt",
                url=url,
            )
            or ""
        )
    except Exception as e:  # trafilatura must never break a scrape
        logger.warning(f"trafilatura extraction failed, falling back to baseline: {e}")
        extracted = ""

    baseline = _boilerplate_stripped_text(html)

    if len(extracted) >= MIN_ABS_CHARS and len(extracted) >= MIN_RATIO * max(len(baseline), 1):
        return extracted
    return baseline


def extract_markdown(html: str, url: str | None = None) -> str | None:
    """Extract main-content as Markdown (LLM/RAG-ready), or ``None``.

    Mirror of :func:`extract_main_text` but emitting Markdown so headings, lists,
    tables and links survive for downstream chunking. Same recall guard: trust
    trafilatura's Markdown only when it retains a meaningful share of the
    boilerplate-stripped baseline, else fall back to a ``markdownify`` pass over
    the chrome-stripped soup. Best-effort — never raises, never fails a scrape.

    Args:
        html: Raw page HTML (for Playwright, the rendered ``page.content()``).
        url: Source URL, passed to trafilatura to improve link resolution.

    Returns:
        Markdown string, or ``None`` for empty input / total extraction failure.
    """
    if not html:
        return None

    try:
        md = (
            trafilatura.extract(
                html,
                include_comments=False,
                include_tables=True,
                include_links=True,
                favor_recall=True,
                output_format="markdown",
                url=url,
            )
            or ""
        )
    except Exception as e:  # trafilatura must never break a scrape
        logger.warning(f"trafilatura markdown extraction failed, falling back: {e}")
        md = ""

    baseline = _boilerplate_stripped_text(html)
    if len(md) >= MIN_ABS_CHARS and len(md) >= MIN_RATIO * max(len(baseline), 1):
        return md

    backstop = _markdownify_stripped(html)
    return backstop or (md or None)


def _stripped_soup(html: str) -> BeautifulSoup | None:
    """Parse ``html`` and decompose site chrome (nav/header/footer/consent).

    Shared by the text baseline and the Markdown backstop so both strip the same
    boilerplate. Never raises: returns ``None`` on parse failure.
    """
    if not html:
        return None

    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception as e:  # lxml unavailable or unparseable input
        logger.warning(f"baseline HTML parse failed: {e}")
        return None

    for tag in soup(list(_BOILERPLATE_TAGS)):
        tag.decompose()
    for tag in soup.select('[role="navigation"]'):
        tag.decompose()
    for selector in _CONSENT_SELECTORS:
        for tag in soup.select(selector):
            tag.decompose()
    return soup


def _markdownify_stripped(html: str) -> str:
    """Markdown of the chrome-stripped soup (recall-guard backstop, never raises)."""
    soup = _stripped_soup(html)
    if soup is None:
        return ""
    try:
        return _html_to_markdown(str(soup), heading_style="ATX").strip()
    except Exception as e:
        logger.warning(f"markdownify backstop failed: {e}")
        return ""


def _boilerplate_stripped_text(html: str) -> str:
    """Visible text with site chrome removed and table rows kept intact.

    Decomposes boilerplate containers, then linearizes every ``<tr>`` as
    ``cell | cell | cell`` (innermost-first, so nested tables survive) before a
    final newline-separated ``get_text``.

    This is the recall-guard *safety net*, so it must never raise: pathological
    HTML that the lxml parser chokes on returns ``""`` (and trafilatura's result,
    if any, still stands) rather than failing the whole scrape.
    """
    soup = _stripped_soup(html)
    if soup is None:
        return ""

    # Reversed document order processes inner rows before the outer rows that
    # contain them, so a replaced inner row's text is folded into the outer cell.
    for row in reversed(soup.find_all("tr")):
        cells = row.find_all(["td", "th"])
        if not cells:
            continue
        joined = " | ".join(c.get_text(separator=" ", strip=True) for c in cells)
        row.replace_with(soup.new_string(f"{joined}\n"))

    text = soup.get_text(separator="\n", strip=True)
    lines = [line for line in text.splitlines() if line.strip()]
    return "\n".join(lines)
