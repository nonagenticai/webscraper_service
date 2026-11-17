"""Relevance scoring for discovered links.

Turns the flat ``links[]`` list every backend already produces into a *ranked*
one, so a caller asking "find this company's pricing and careers pages" gets one
call instead of a scrape-then-guess loop. Shared by all three link extractors
(httpx, Playwright, Scrapy) through the single conversion seam in
``routers.scrape._convert_links``, and reused by the Scrapy backend to order its
crawl frontier (best-first instead of blind BFS).

Adapted from crawl4ai's link scoring (``crawl4ai/link_preview.py`` and
``utils.calculate_total_score``) at commit ``7e80152``, © UncleCode, Apache-2.0
— see THIRD_PARTY_NOTICES.md. The adaptation drops their LLM/embedding
contextual scorer and their config objects; relevance here is deliberately
dependency-free token overlap so the service adds no new packages for this
feature.

Scoring model
-------------
Every link gets an *intrinsic* score in ``[0, 1]`` from signals that need no
caller input: anchor-text quality, URL path depth, query-string noise,
same-domain, ``rel=nofollow``, and non-HTML file extensions.

When the caller supplies a query, a *contextual* score in ``[0, 1]`` measures how
well the anchor text and URL slug match it, and the final score is the mean of
the two. Without a query the intrinsic score stands alone. This keeps the
no-query case meaningful (nav/footer junk still sorts below body links) instead
of collapsing to a constant.

Scores are advisory. They are additive on ``LinkInfo.score`` and never reorder
or drop links unless the caller opts in via ``sort_links_by_score`` — the
``links[]`` dict shape is a load-bearing contract for downstream consumers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import unquote, urlsplit

# Anchor text shorter than this carries almost no signal ("›", "1", "»").
_MIN_USEFUL_TEXT = 3
# Beyond this an anchor is usually a sentence or a whole card, not a label.
_MAX_USEFUL_TEXT = 60

# Generic anchors that are structurally links but semantically empty. Matched on
# the whole normalized text, never as a substring, so "Read more about pricing"
# is not penalized.
_LOW_VALUE_ANCHORS = frozenset(
    {
        "click here",
        "here",
        "read more",
        "more",
        "learn more",
        "continue",
        "continue reading",
        "next",
        "previous",
        "prev",
        "back",
        "top",
        "link",
        "this",
        "view",
        "see more",
        "details",
    }
)

# Tracking/session parameters contribute noise, not identity.
_NOISE_QUERY_PARAMS = ("utm_", "fbclid", "gclid", "msclkid", "mc_cid", "mc_eid", "ref", "sessionid")

# Non-HTML targets. Mildly penalized rather than excluded: a PDF is a legitimate
# scrape target (discovery-engine fetches publisher full-text PDFs), just rarely
# the *most* relevant link on a page.
_NON_HTML_EXTENSIONS = frozenset(
    {
        "pdf", "doc", "docx", "xls", "xlsx", "ppt", "pptx", "zip", "gz", "tar", "rar",
        "jpg", "jpeg", "png", "gif", "svg", "webp", "ico", "bmp", "tiff",
        "mp3", "mp4", "avi", "mov", "wmv", "webm", "wav", "ogg",
        "css", "js", "json", "xml", "rss", "atom", "exe", "dmg", "pkg",
    }
)  # fmt: skip

# Dropped from queries and link text alike — they match everything and so
# discriminate nothing.
_STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "has",
        "have", "how", "in", "is", "it", "its", "of", "on", "or", "our", "that", "the",
        "their", "this", "to", "was", "we", "were", "what", "when", "where", "which",
        "who", "will", "with", "you", "your",
    }
)  # fmt: skip

# Two tokens match loosely when they share a common prefix of at least this many
# characters, so "career"/"careers" match without a stemmer dependency. Note a
# strict ``startswith`` is NOT enough: "price" does not prefix "pricing" (they
# diverge at "pric|e" vs "pric|ing"), which is precisely the morphology callers
# hit most.
_PREFIX_MATCH_MIN = 4

# ...and the shared prefix must also cover this fraction of the shorter token, so
# "serv" does not marry "service" to "servers".
_PREFIX_MATCH_COVERAGE = 0.6

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Intrinsic signal weights. Normalized at use, so these read as relative
# importance rather than as a sum that must be maintained by hand.
_INTRINSIC_WEIGHTS = {
    "text_quality": 0.30,
    "path_depth": 0.20,
    "query_noise": 0.10,
    "same_domain": 0.20,
    "followable": 0.10,
    "html_target": 0.10,
}


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens with stopwords and 1-char noise removed."""
    if not text:
        return []
    return [t for t in _TOKEN_RE.findall(text.lower()) if len(t) > 1 and t not in _STOPWORDS]


def _common_prefix_len(a: str, b: str) -> int:
    """Number of leading characters two strings share."""
    n = 0
    for ca, cb in zip(a, b, strict=False):
        if ca != cb:
            break
        n += 1
    return n


def _tokens_match(a: str, b: str) -> bool:
    """True when two tokens are equal or share a long-enough common prefix.

    Deliberately a *common prefix* test, not ``startswith``: "price" and
    "pricing" share "pric" but neither prefixes the other, and that pair is the
    single most common query/slug mismatch in practice.
    """
    if a == b:
        return True
    shorter = a if len(a) <= len(b) else b
    required = max(_PREFIX_MATCH_MIN, round(_PREFIX_MATCH_COVERAGE * len(shorter)))
    return _common_prefix_len(a, b) >= required


def _text_quality(text: str) -> float:
    """Score anchor text on informativeness in ``[0, 1]``."""
    normalized = " ".join(text.split()).lower()
    if not normalized:
        return 0.0
    if normalized in _LOW_VALUE_ANCHORS:
        return 0.1
    if normalized.isdigit():
        return 0.1

    length = len(normalized)
    if length < _MIN_USEFUL_TEXT:
        return 0.2
    if length <= _MAX_USEFUL_TEXT:
        return 1.0
    # Long anchors (whole sentences / card bodies) decay rather than cliff.
    return max(0.3, _MAX_USEFUL_TEXT / length)


def _path_depth_score(path: str) -> float:
    """Shallower paths score higher; ``/pricing`` beats ``/a/b/c/d/e``."""
    segments = [s for s in path.split("/") if s]
    if not segments:
        return 0.9  # A bare root link is useful but rarely the answer.
    return max(0.1, 1.0 - 0.15 * (len(segments) - 1))


def _query_noise_score(query: str) -> float:
    """Fewer, cleaner query parameters score higher."""
    if not query:
        return 1.0
    params = [p for p in query.split("&") if p]
    lowered = query.lower()
    noise = sum(1 for marker in _NOISE_QUERY_PARAMS if marker in lowered)
    return max(0.1, 1.0 - 0.15 * len(params) - 0.15 * noise)


def _html_target_score(path: str) -> float:
    """Penalize obvious non-HTML targets without excluding them."""
    _, _, tail = path.rpartition("/")
    _, dot, ext = tail.rpartition(".")
    if not dot:
        return 1.0
    return 0.4 if ext.lower() in _NON_HTML_EXTENSIONS else 1.0


def _contextual_score(query_tokens: list[str], text: str, path: str) -> float:
    """Fraction of query tokens found in the anchor text or URL slug.

    Slug matches are worth slightly more than anchor-text matches: a URL segment
    is authored by the site owner and is a stronger topical claim than link
    wording, which is often generic.
    """
    if not query_tokens:
        return 0.0

    text_tokens = tokenize(text)
    slug_tokens = tokenize(unquote(path).replace("-", " ").replace("_", " "))

    total = 0.0
    for q in query_tokens:
        if any(_tokens_match(q, s) for s in slug_tokens):
            total += 1.0
        elif any(_tokens_match(q, t) for t in text_tokens):
            total += 0.8
    return min(1.0, total / len(query_tokens))


@dataclass(frozen=True)
class LinkScorer:
    """Reusable scorer bound to one page's base domain and caller query.

    Built once per scrape so the query is tokenized once rather than per link —
    a page can carry thousands of anchors.
    """

    query_tokens: tuple[str, ...] = ()

    @classmethod
    def for_query(cls, query: str | None) -> LinkScorer:
        """Build a scorer for an optional caller query."""
        return cls(query_tokens=tuple(tokenize(query or "")))

    @property
    def has_query(self) -> bool:
        return bool(self.query_tokens)

    def score(
        self,
        url: str,
        text: str = "",
        rel: str | None = None,
        link_type: str | None = None,
    ) -> float:
        """Score one link in ``[0, 1]``, rounded to 4 decimals.

        Args:
            url: Absolute link target.
            text: Anchor text.
            rel: The ``rel`` attribute value (may hold several tokens).
            link_type: ``"internal"``/``"external"`` as classified by the
                backend. Callers that do not classify may pass ``None``, which
                scores the same-domain signal neutrally rather than guessing.

        Returns:
            The relevance score. Higher is more relevant.
        """
        if not url:
            return 0.0

        parts = urlsplit(url)
        signals = {
            "text_quality": _text_quality(text),
            "path_depth": _path_depth_score(parts.path),
            "query_noise": _query_noise_score(parts.query),
            "same_domain": _same_domain_score(link_type),
            "followable": 0.2 if rel and "nofollow" in rel.lower() else 1.0,
            "html_target": _html_target_score(parts.path),
        }
        total_weight = sum(_INTRINSIC_WEIGHTS.values())
        intrinsic = sum(_INTRINSIC_WEIGHTS[k] * v for k, v in signals.items()) / total_weight

        if not self.query_tokens:
            return round(intrinsic, 4)

        contextual = _contextual_score(list(self.query_tokens), text, parts.path)
        return round(0.5 * intrinsic + 0.5 * contextual, 4)


def _same_domain_score(link_type: str | None) -> float:
    """Internal links score highest; unclassified links score neutrally."""
    if link_type == "internal":
        return 1.0
    if link_type == "external":
        return 0.4
    return 0.7


__all__ = ["LinkScorer", "tokenize"]
