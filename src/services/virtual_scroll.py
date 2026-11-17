"""Infinite-scroll and virtualized-list handling for the Playwright backend.

Without this, a scrape of an infinite feed returns whatever happened to be in
the first viewport — a fraction of the page, reported as a complete success.

Two distinct page behaviours have to be handled, and conflating them is how this
feature silently half-works:

*Append* lists (most product grids, search results, news feeds) add nodes to the
container as you scroll. Scrolling alone is enough — the DOM ends up holding
everything.

*Virtualized* lists (Twitter/X-style feeds, big data grids) REMOVE off-screen
nodes as they add new ones, to keep the node count flat. Scrolling alone leaves
only the final window in the DOM, so a naive implementation would return *less*
than not scrolling at all on exactly the pages that need it most. So each step
also captures the container's direct children, and if the accumulated set turns
out larger than what survived in the DOM, the container is rewritten with the
full de-duplicated set before content extraction.

Rewriting the live DOM means everything downstream — ``html_content``,
``text_content``, ``markdown_content``, ``links``, ``images`` and
``extraction_schema`` — sees the merged page with no further plumbing.

Adapted in concept from crawl4ai's ``VirtualScrollConfig``, Apache-2.0 — see
THIRD_PARTY_NOTICES.md. The scroll loop is driven from Python rather than a
single long in-page script so each step stays bounded by Playwright's own
timeout and the scrape remains cancellable.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from playwright.async_api import Page

    from schemas.scrape import VirtualScrollConfig

logger = logging.getLogger(__name__)

# Two consecutive steps that add neither height nor new children mean the feed
# is exhausted (or was never infinite). One is too twitchy — a slow XHR often
# lands during the following step.
_IDLE_STEPS_BEFORE_STOP = 2

# Capturing children is only meaningful for an explicit container. Guard the
# total we will hold so a runaway feed cannot exhaust pod memory.
MAX_CAPTURED_CHILDREN = 5_000

# Distinguishes "the in-page script failed" from "the script ran and returned
# null" (container selector matched nothing). Reporting the first as the second
# sent callers hunting a selector bug that was really a detached frame or CSP.
_EVAL_FAILED = object()

# Read the scroll position/height of the container, or of the page when no
# container selector was given.
_METRICS_JS = """
(selector) => {
  const el = selector ? document.querySelector(selector) : null;
  if (selector && !el) return null;
  const target = el || document.scrollingElement || document.body;
  return {
    scrollHeight: target.scrollHeight,
    childCount: el ? el.children.length : 0,
  };
}
"""

_SCROLL_JS = """
([selector, amount]) => {
  const el = selector ? document.querySelector(selector) : null;
  if (selector && !el) return false;
  const target = el || document.scrollingElement || document.body;
  const step = amount > 0 ? amount : target.clientHeight || window.innerHeight;
  if (el) {
    el.scrollTop = el.scrollTop + step;
  } else {
    window.scrollBy(0, step);
  }
  return true;
}
"""

# Each child is returned with a STABLE identity key, not just its markup.
# De-duplicating on outerHTML alone counted one element twice the moment
# anything about it changed between steps — a lazy <img> resolving its src, an
# IntersectionObserver toggling a class, a relative timestamp ticking over —
# and the duplicate then made an ordinary append list look like a recycling one.
# Text alone is not identity either: an image-only masonry grid of
# `<div><img></div>` tiles carries none, so every tile collapsed onto one key and
# the accumulator held a single entry. The signature therefore also folds in the
# child's first few media/link targets, which is what distinguishes those tiles.
# A child that yields NO signal at all is reported as unkeyable rather than
# silently overwriting its siblings.
_CHILDREN_JS = """
(selector) => {
  const el = document.querySelector(selector);
  if (!el) return [];
  return Array.from(el.children).map((c) => {
    const explicit = c.id
      || c.getAttribute('data-key')
      || c.getAttribute('data-id')
      || c.getAttribute('data-item-id')
      || c.getAttribute('data-testid');
    if (explicit) return { key: 'k:' + explicit, html: c.outerHTML };
    const text = (c.textContent || '').replace(/\\s+/g, ' ').trim().slice(0, 200);
    const refs = Array.from(c.querySelectorAll('img[src],source[srcset],a[href],[data-src]'))
      .slice(0, 5)
      .map((n) => n.getAttribute('src') || n.getAttribute('srcset')
                 || n.getAttribute('href') || n.getAttribute('data-src') || '')
      .join('|')
      .slice(0, 300);
    if (!text && !refs) return { key: '', html: c.outerHTML };
    return { key: 't:' + c.tagName + '|' + text + '|' + refs, html: c.outerHTML };
  });
}
"""

_REWRITE_JS = """
([selector, html]) => {
  const el = document.querySelector(selector);
  if (!el) return false;
  el.innerHTML = html;
  return true;
}
"""


async def apply_virtual_scroll(
    page: Page,
    config: VirtualScrollConfig,
    budget_seconds: float | None = None,
) -> dict[str, Any]:
    """Scroll a page (or a container within it) to materialize lazy content.

    Never raises: scrolling is an enhancement, and a page that resists it must
    still return whatever content it has rather than failing the whole scrape.

    Args:
        page: The live Playwright page, already navigated.
        config: The caller's scroll configuration.
        budget_seconds: Wall-clock the scroll may spend. The caller's scrape has
            one overall deadline; overrunning it discards the *entire* page, so
            scrolling stops and reports ``stop_reason="budget"`` rather than
            taking the whole result down with it. ``None`` means unbounded
            (tests and callers that have already reserved the time).

    Returns:
        A stats dict recorded into the scrape metadata:
        ``steps``, ``captured``, ``merged``, ``stopped_early``, ``truncated``
        and ``stop_reason`` — one of ``step_cap`` (ran every requested step,
        the feed probably has more), ``exhausted`` (two idle steps, the feed
        ended), ``budget`` (out of time), ``scroll_failed`` (the page refused
        to scroll) or ``capture_cap`` (hit ``MAX_CAPTURED_CHILDREN``).
    """
    selector = config.container_selector
    stats: dict[str, Any] = {
        "steps": 0,
        "captured": 0,
        "merged": False,
        "stopped_early": False,
        "truncated": False,
        "unkeyable": False,
        "stop_reason": "step_cap",
    }

    deadline = None if budget_seconds is None else time.monotonic() + budget_seconds

    # Ordered de-duplication keyed on each child's STABLE identity; the value is
    # the freshest markup seen for it, so a lazily-resolved <img src> or a
    # hydrated class lands in the merged DOM instead of the placeholder.
    captured: dict[str, str] = {}
    idle_steps = 0
    last_height = -1

    for step in range(config.scroll_count):
        if deadline is not None and time.monotonic() >= deadline:
            stats["stopped_early"] = True
            stats["stop_reason"] = "budget"
            break

        metrics = await _safe_eval(page, _METRICS_JS, selector)
        if metrics is _EVAL_FAILED or metrics is None:
            if metrics is _EVAL_FAILED:
                # The page, not the selector: a detached frame, a navigation
                # mid-scroll, a CSP block. Naming it container_not_found sent
                # callers to debug a selector that was fine.
                logger.warning("virtual_scroll page evaluation failed for selector %r", selector)
                stats["error"] = "page_eval_failed"
            else:
                # The container selector matched nothing. Say so rather than
                # silently scrolling the window and reporting success.
                logger.warning("virtual_scroll container %r not found; skipping", selector)
                stats["error"] = "container_not_found"
            stats["stopped_early"] = True
            stats["stop_reason"] = "scroll_failed"
            # Break rather than return: a frame that detaches on step 9 of 10
            # must not throw away the nine steps already captured and report
            # captured: 0, which reads as "this container was always empty".
            break

        before = len(captured)
        if selector:
            _absorb_children(captured, await _safe_children(page, selector), stats)

        height = metrics.get("scrollHeight", 0)
        # A virtualized list keeps a CONSTANT scrollHeight (fixed spacer, window
        # of recycled nodes) — height alone would call the feed exhausted after
        # three steps on exactly the pages this module exists for. New children
        # are growth too.
        grew = height > last_height or len(captured) > before
        last_height = max(last_height, height)

        scrolled = await _safe_eval(page, _SCROLL_JS, [selector, config.scroll_by])
        if scrolled is _EVAL_FAILED or not scrolled:
            stats["stopped_early"] = True
            stats["stop_reason"] = "scroll_failed"
            break
        await page.wait_for_timeout(config.wait_ms)
        stats["steps"] = step + 1

        if stats["truncated"]:
            stats["stopped_early"] = True
            stats["stop_reason"] = "capture_cap"
            break

        if grew:
            idle_steps = 0
        else:
            idle_steps += 1
            if idle_steps >= _IDLE_STEPS_BEFORE_STOP:
                stats["stopped_early"] = True
                stats["stop_reason"] = "exhausted"
                break

    # Final sweep: the last scroll's content has not been captured yet.
    if selector:
        _absorb_children(captured, await _safe_children(page, selector), stats)

    stats["captured"] = len(captured)

    # The rewrite REPLACES the live container, so it is only ever safe when the
    # accumulator is a superset of what is on the page. Two ways it isn't:
    #   - truncated: the cap stopped absorption mid-batch, so children standing
    #     in the DOM right now were never captured. Rewriting would DELETE them —
    #     the last screenful of a huge feed, gone, on a "completed" response.
    #   - unkeyable: at least one child produced no identity signal, so distinct
    #     children may have collapsed onto one entry and the merged set is short.
    # In both cases the honest move is to leave the DOM alone and say why.
    if selector and captured and not stats["truncated"] and not stats.get("unkeyable"):
        live = await _safe_eval(page, _METRICS_JS, selector)
        child_count = live.get("childCount", 0) if isinstance(live, dict) else 0
        # More captured than survive in the DOM ⇒ the list recycles nodes, so
        # the DOM alone would under-report. Rewrite it with the full set. With
        # identity-keyed capture an append list yields captured == childCount,
        # so an ordinary grid is no longer rewritten (and no longer duplicated).
        if len(captured) > child_count:
            merged = "".join(captured.values())
            stats["merged"] = _safe_eval_bool(
                await _safe_eval(page, _REWRITE_JS, [selector, merged])
            )

    return stats


def _absorb_children(captured: dict[str, str], children: list[Any], stats: dict[str, Any]) -> None:
    """Merge a capture batch into the accumulator, enforcing the hard cap.

    The cap used to be tested only *before* a batch, so a single step could add
    an unbounded number of children past it — the guard read like a bound and
    was not one.

    Identity is per-batch disambiguated: two children of one container CAN share
    a key (a duplicated ``id`` in generated markup, two rows with identical
    text), and letting the second overwrite the first drops a real item. Within
    one batch each repeat gets its own slot; across batches the Nth occurrence
    of a key maps to the same slot, which is what lets a recycling list
    accumulate.
    """
    seen_in_batch: dict[str, int] = {}
    for child in children:
        if not isinstance(child, dict):
            continue
        key = child.get("key")
        html = child.get("html")
        if not isinstance(key, str) or not isinstance(html, str):
            continue
        if not key:
            # No id, no text, no media: nothing distinguishes this child from
            # its siblings, so the accumulator cannot be trusted as a superset
            # of the DOM and the merge is suppressed.
            stats["unkeyable"] = True
            continue
        occurrence = seen_in_batch.get(key, 0)
        seen_in_batch[key] = occurrence + 1
        slot = key if occurrence == 0 else f"{key}#{occurrence}"
        if slot not in captured and len(captured) >= MAX_CAPTURED_CHILDREN:
            stats["truncated"] = True
            return
        captured[slot] = html


async def _safe_children(page: Page, selector: str) -> list[Any]:
    """Capture the container's children, treating any failure as an empty batch."""
    result = await _safe_eval(page, _CHILDREN_JS, selector)
    return result if isinstance(result, list) else []


def _safe_eval_bool(value: Any) -> bool:
    """A failed evaluation is not a successful merge."""
    return value is not _EVAL_FAILED and bool(value)


async def _safe_eval(page: Page, script: str, arg: Any) -> Any:
    """Evaluate in-page JS, downgrading any failure to ``_EVAL_FAILED``.

    A detached frame, a navigation mid-scroll or a CSP quirk must not turn a
    working scrape into a 500 — but it must also not be mistaken for the script
    legitimately returning ``null``.
    """
    try:
        return await page.evaluate(script, arg)
    except Exception as e:
        logger.debug("virtual_scroll step failed: %s", e)
        return _EVAL_FAILED


__all__ = ["MAX_CAPTURED_CHILDREN", "apply_virtual_scroll"]
