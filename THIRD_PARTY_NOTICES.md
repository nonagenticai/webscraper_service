# Third-party notices

This service vendors and adapts a small amount of third-party source. Each entry
records what was taken, from where, and under which licence. Vendored code is
adapted to this repo's conventions (type hints, ruff config, Python 3.14) rather
than copied verbatim, and the corresponding module carries a header pointing
back here.

---

## crawl4ai

- **Upstream:** https://github.com/unclecode/crawl4ai
- **Commit referenced:** `7e801521428ee12509994d39151006f64055ebe3` (2026-07-15, v0.9.2)
- **Licence:** Apache License 2.0 — https://www.apache.org/licenses/LICENSE-2.0
- **Copyright:** © UncleCode (unclecode@kidocode.com)

crawl4ai is **not** a dependency of this service and is not installed. Specific
algorithms were adapted into our own modules:

| Our module | Adapted from | What was taken |
|---|---|---|
| `src/services/link_scoring.py` | `crawl4ai/link_preview.py`, `crawl4ai/utils.py` (`calculate_total_score`) | The link-relevance scoring model: intrinsic link-quality signals plus a contextual query-match term. Their LLM/embedding contextual scorer and config objects were dropped; relevance here is dependency-free token overlap. |
| `src/services/virtual_scroll.py` | `crawl4ai/async_configs.py` (`VirtualScrollConfig`) and the associated scroll handling | The concept of scrolling a container while accumulating its children, so *virtualized* lists that recycle DOM nodes are captured in full. The implementation is our own: the loop is driven from Python (bounded and cancellable per step) rather than one long in-page script, and the merged children are written back into the live DOM so all downstream extraction sees the whole page. |
| `src/services/extraction_schema.py` | `crawl4ai/extraction_strategy.py` (`JsonElementExtractionStrategy`, `JsonCssExtractionStrategy`) | The declarative CSS extraction model and its schema dialect, which we keep wire-compatible (`baseSelector` is accepted verbatim). The `computed`, `regex` and `source` field types were dropped, as was the entire LLM extraction path (`LLMExtractionStrategy` and its prompts) — see the module docstring for the reasoning. |

### Why adapted rather than depended on

`crawl4ai==0.9.2` declares `unclecode-litellm==1.81.13` — an exact-pinned fork of
litellm — as a mandatory top-level dependency. Taking it would ship an LLM client
into a scraper (this platform routes all LLM traffic through the workflow
manager), add an unpatchable pin under our `pip-audit` gate, and pull a large
transitive tree into an image already close to its memory limit. Upstream also
classifies only up to Python 3.13, while this service requires 3.14.

### Apache-2.0 obligations

The Apache 2.0 licence permits this use. Per §4 we retain attribution (this file),
state that changes were made (above), and note that the adapted portions remain
subject to the upstream copyright and licence. No NOTICE file is distributed by
the upstream project.
