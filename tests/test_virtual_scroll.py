"""Tests for infinite-scroll / virtualized-list handling (services.virtual_scroll)."""

from typing import Any

import pytest

from schemas.scrape import VirtualScrollConfig
from services.virtual_scroll import (
    MAX_CAPTURED_CHILDREN,
    apply_virtual_scroll,
)


class FakePage:
    """A scriptable stand-in for a Playwright Page.

    Models the two behaviours that matter: an *append* list where children
    accumulate in the DOM, and a *virtualized* list where old children are
    evicted as new ones arrive.
    """

    def __init__(
        self,
        *,
        pages_of_children: list[list[str]] | None = None,
        container_exists: bool = True,
        window_pages: int = 0,
        recycle: bool = False,
        fail_on: str | None = None,
        constant_height: bool = False,
        mutate_html: bool = False,
    ):
        self.pages_of_children = pages_of_children or []
        self.container_exists = container_exists
        self.window_pages = window_pages
        self.recycle = recycle
        self.fail_on = fail_on
        # A real virtualized list keeps a FIXED spacer height while recycling
        # its window of nodes, so scrollHeight never changes. Modelling height
        # as always-growing is what hid the height-only termination bug.
        self.constant_height = constant_height
        # Re-render each child's markup on every step (lazy <img src> resolving,
        # an IntersectionObserver toggling a class) while its identity is stable.
        self.mutate_html = mutate_html
        self.scroll_index = 0
        self.waits: list[int] = []
        self.rewritten: str | None = None
        self.evaluations: list[str] = []

    # -- DOM model ---------------------------------------------------------
    def _visible_children(self) -> list[str]:
        if not self.pages_of_children:
            return []
        idx = min(self.scroll_index, len(self.pages_of_children) - 1)
        if self.recycle:
            return self.pages_of_children[idx]
        merged: list[str] = []
        for chunk in self.pages_of_children[: idx + 1]:
            merged.extend(chunk)
        return merged

    def _child_records(self) -> list[dict[str, str]]:
        """Mirror _CHILDREN_JS: stable identity key + current markup."""
        records = []
        for child in self._visible_children():
            html = child
            if self.mutate_html:
                # Same element, different markup on each read.
                html = child.replace("<li", f"<li data-render={self.scroll_index}")
            records.append({"key": f"t:LI|{child}", "html": html})
        return records

    def _height(self) -> int:
        if self.constant_height:
            return 100_000
        if self.pages_of_children:
            return 1000 * (min(self.scroll_index, len(self.pages_of_children) - 1) + 1)
        return 1000 * (min(self.scroll_index, max(self.window_pages - 1, 0)) + 1)

    # -- Playwright surface ------------------------------------------------
    async def evaluate(self, script: str, arg: Any = None):
        if self.fail_on and self.fail_on in script:
            raise RuntimeError("boom")
        self.evaluations.append(script)

        if "scrollHeight" in script:
            if arg and not self.container_exists:
                return None
            return {"scrollHeight": self._height(), "childCount": len(self._visible_children())}
        if "Array.from" in script:
            return self._child_records()
        if "scrollTop" in script:
            selector = arg[0] if isinstance(arg, list) else arg
            if selector and not self.container_exists:
                return False
            self.scroll_index += 1
            return True
        if "innerHTML = html" in script:
            self.rewritten = arg[1]
            return True
        raise AssertionError(f"unexpected script: {script}")

    async def wait_for_timeout(self, ms: int) -> None:
        self.waits.append(ms)


def _cfg(**kw) -> VirtualScrollConfig:
    return VirtualScrollConfig(**{"container_selector": "#feed", "wait_ms": 50, **kw})


class TestAppendList:
    """The common case: nodes accumulate, so scrolling alone is enough."""

    @pytest.mark.asyncio
    async def test_scrolls_and_waits_each_step(self):
        page = FakePage(pages_of_children=[["<li>1</li>"], ["<li>2</li>"], ["<li>3</li>"]])
        stats = await apply_virtual_scroll(page, _cfg(scroll_count=3))
        assert stats["steps"] == 3
        assert page.waits == [50, 50, 50]

    @pytest.mark.asyncio
    async def test_does_not_rewrite_the_dom_when_nothing_was_lost(self):
        """An append list already holds everything — rewriting would be churn."""
        page = FakePage(pages_of_children=[["<li>1</li>"], ["<li>2</li>"]])
        stats = await apply_virtual_scroll(page, _cfg(scroll_count=2))
        assert stats["merged"] is False
        assert page.rewritten is None

    @pytest.mark.asyncio
    async def test_children_whose_markup_changes_between_steps_are_not_duplicated(self):
        """Regression: identity is the element, not its current markup.

        De-duplicating on outerHTML counted one card twice the moment a lazy
        <img src> resolved or a class was toggled between steps. captured then
        exceeded the live child count, the recycle heuristic fired on an
        ordinary append grid, and the container was REWRITTEN with duplicates —
        every downstream field (links, images, text, extracted_data) doubled,
        reported as a clean 200.
        """
        page = FakePage(
            pages_of_children=[["<li>a</li>", "<li>b</li>"], ["<li>c</li>"]],
            mutate_html=True,
        )
        stats = await apply_virtual_scroll(page, _cfg(scroll_count=3))
        assert stats["captured"] == 3
        assert stats["merged"] is False
        assert page.rewritten is None


class TestVirtualizedList:
    """The case that makes naive scrolling WORSE than not scrolling."""

    @pytest.mark.asyncio
    async def test_recycled_children_are_accumulated_and_merged_back(self):
        page = FakePage(
            pages_of_children=[["<li>1</li>", "<li>2</li>"], ["<li>3</li>"], ["<li>4</li>"]],
            recycle=True,
        )
        stats = await apply_virtual_scroll(page, _cfg(scroll_count=3))
        assert stats["captured"] == 4
        assert stats["merged"] is True
        # Every item survives, in first-seen order, exactly once.
        assert page.rewritten == "<li>1</li><li>2</li><li>3</li><li>4</li>"

    @pytest.mark.asyncio
    async def test_duplicate_children_across_steps_are_deduplicated(self):
        page = FakePage(
            pages_of_children=[["<li>a</li>"], ["<li>a</li>", "<li>b</li>"], ["<li>b</li>"]],
            recycle=True,
        )
        stats = await apply_virtual_scroll(page, _cfg(scroll_count=3))
        assert stats["captured"] == 2
        assert page.rewritten == "<li>a</li><li>b</li>"

    @pytest.mark.asyncio
    async def test_final_window_is_captured_after_the_last_scroll(self):
        """Without a post-loop sweep the last screenful is scrolled past and lost."""
        page = FakePage(
            pages_of_children=[["<li>1</li>"], ["<li>2</li>"], ["<li>last</li>"]],
            recycle=True,
        )
        await apply_virtual_scroll(page, _cfg(scroll_count=2))
        assert "<li>last</li>" in (page.rewritten or "")


class TestTermination:
    @pytest.mark.asyncio
    async def test_stops_early_once_the_page_stops_growing(self):
        page = FakePage(pages_of_children=[["<li>1</li>"], ["<li>2</li>"]])
        stats = await apply_virtual_scroll(page, _cfg(scroll_count=20))
        assert stats["stopped_early"] is True
        assert stats["steps"] < 20

    @pytest.mark.asyncio
    async def test_respects_the_step_cap_on_an_endless_feed(self):
        page = FakePage(pages_of_children=[[f"<li>{i}</li>"] for i in range(500)])
        stats = await apply_virtual_scroll(page, _cfg(scroll_count=5))
        assert stats["steps"] == 5
        assert stats["stopped_early"] is False

    @pytest.mark.asyncio
    async def test_captured_children_are_bounded(self):
        """The cap must bound the accumulator, not merely be consulted.

        It used to be tested only BEFORE each batch, so a single step could
        push it arbitrarily far past the limit — the guard read like a bound
        and was not one.
        """
        page = FakePage(
            pages_of_children=[
                [f"<li>{i}</li>" for i in range(MAX_CAPTURED_CHILDREN + 100)],
                ["<li>extra</li>"],
            ],
            recycle=True,
        )
        stats = await apply_virtual_scroll(page, _cfg(scroll_count=2))
        assert stats["captured"] == MAX_CAPTURED_CHILDREN
        assert stats["truncated"] is True
        assert stats["stop_reason"] == "capture_cap"

    @pytest.mark.asyncio
    async def test_virtualized_feed_with_constant_height_keeps_scrolling(self):
        """Regression: the feature's whole reason to exist.

        A virtualized list holds a fixed spacer height and recycles its nodes,
        so scrollHeight NEVER changes. Terminating on height alone declared the
        feed exhausted after three steps and reported that as a clean stop,
        losing ~90% of a 10-step feed with nothing in the response to show it.
        """
        page = FakePage(
            pages_of_children=[[f"<li>{i}</li>"] for i in range(10)],
            recycle=True,
            constant_height=True,
        )
        stats = await apply_virtual_scroll(page, _cfg(scroll_count=10))
        assert stats["steps"] == 10
        assert stats["stopped_early"] is False
        assert stats["captured"] == 10

    @pytest.mark.asyncio
    async def test_a_constant_height_feed_that_stops_producing_still_terminates(self):
        """The new growth signal must not become a never-stop signal."""
        page = FakePage(
            pages_of_children=[["<li>1</li>"], ["<li>2</li>"]],
            recycle=True,
            constant_height=True,
        )
        stats = await apply_virtual_scroll(page, _cfg(scroll_count=20))
        assert stats["stopped_early"] is True
        assert stats["stop_reason"] == "exhausted"
        assert stats["steps"] < 20


class TestFailureHandling:
    @pytest.mark.asyncio
    async def test_missing_container_reports_an_error_instead_of_scrolling_the_window(self):
        """Silently scrolling the window would report success for a no-op."""
        page = FakePage(container_exists=False)
        stats = await apply_virtual_scroll(page, _cfg(scroll_count=3))
        assert stats["error"] == "container_not_found"
        assert stats["steps"] == 0

    @pytest.mark.asyncio
    async def test_in_page_failure_does_not_raise(self):
        page = FakePage(pages_of_children=[["<li>1</li>"]], fail_on="scrollHeight")
        stats = await apply_virtual_scroll(page, _cfg(scroll_count=2))
        assert stats["steps"] == 0

    @pytest.mark.asyncio
    async def test_page_eval_failure_is_not_reported_as_a_missing_container(self):
        """A detached frame and a wrong selector are different bugs to chase."""
        page = FakePage(pages_of_children=[["<li>1</li>"]], fail_on="scrollHeight")
        stats = await apply_virtual_scroll(page, _cfg(scroll_count=2))
        assert stats["error"] == "page_eval_failed"

    @pytest.mark.asyncio
    async def test_a_page_that_refuses_to_scroll_says_so(self):
        """Breaking out silently was indistinguishable from a feed that ended."""

        class NoScrollPage(FakePage):
            async def evaluate(self, script: str, arg: Any = None):
                if "scrollTop" in script:
                    return False
                return await super().evaluate(script, arg)

        page = NoScrollPage(pages_of_children=[["<li>1</li>"], ["<li>2</li>"]])
        stats = await apply_virtual_scroll(page, _cfg(scroll_count=5))
        assert stats["stop_reason"] == "scroll_failed"
        assert stats["stopped_early"] is True

    @pytest.mark.asyncio
    async def test_scrolling_stops_when_the_request_budget_is_spent(self):
        """30 steps x 5s of wait is a VALID config that outruns the scrape.

        Overrunning the scrape's single deadline raises, discarding the entire
        page — so the scroll gives up its remaining steps instead of taking the
        result down with it.
        """
        page = FakePage(pages_of_children=[[f"<li>{i}</li>"] for i in range(30)])
        stats = await apply_virtual_scroll(page, _cfg(scroll_count=30), budget_seconds=0.0)
        assert stats["steps"] == 0
        assert stats["stop_reason"] == "budget"
        assert stats["stopped_early"] is True

    @pytest.mark.asyncio
    async def test_window_scrolling_without_a_container_selector(self):
        page = FakePage(window_pages=4)
        stats = await apply_virtual_scroll(page, VirtualScrollConfig(scroll_count=3, wait_ms=50))
        assert stats["steps"] == 3
        # No container ⇒ nothing to accumulate or merge.
        assert stats["captured"] == 0
        assert stats["merged"] is False


class TestConfigValidation:
    def test_scroll_count_is_bounded(self):
        import pydantic

        with pytest.raises(pydantic.ValidationError):
            VirtualScrollConfig(scroll_count=31)
        with pytest.raises(pydantic.ValidationError):
            VirtualScrollConfig(scroll_count=0)

    def test_wait_ms_is_bounded(self):
        import pydantic

        with pytest.raises(pydantic.ValidationError):
            VirtualScrollConfig(wait_ms=10)

    def test_defaults_are_sane(self):
        cfg = VirtualScrollConfig()
        assert cfg.container_selector is None
        assert cfg.scroll_by == 0  # one viewport height
        assert 1 <= cfg.scroll_count <= 30


class TestMergeSafety:
    """The rewrite REPLACES the live container, so it must only ever run when
    the accumulator is a superset of what is on the page."""

    @pytest.mark.asyncio
    async def test_a_truncated_capture_never_rewrites_the_container(self):
        """Regression: the cap stops absorption mid-batch, so children standing
        in the DOM were never captured — rewriting DELETED them, on a
        `completed` response."""
        page = FakePage(
            pages_of_children=[
                [f"<li>{i}</li>" for i in range(MAX_CAPTURED_CHILDREN + 100)],
                ["<li>fresh-A</li>", "<li>fresh-B</li>"],
            ],
            recycle=True,
        )
        stats = await apply_virtual_scroll(page, _cfg(scroll_count=2))
        assert stats["truncated"] is True
        assert stats["merged"] is False
        assert page.rewritten is None

    @pytest.mark.asyncio
    async def test_image_only_tiles_are_distinguished_by_their_media(self):
        """Regression: a masonry grid of image tiles carries no text, so every
        tile keyed identically and the accumulator held ONE entry — the feature
        silently no-opped on exactly the pages it exists for."""

        class TilePage(FakePage):
            def _child_records(self):
                return [
                    {
                        "key": f"t:DIV||https://cdn.example/{c}.jpg",
                        "html": f"<div><img src='{c}'></div>",
                    }
                    for c in self._visible_children()
                ]

        page = TilePage(
            pages_of_children=[["a", "b"], ["c", "d"], ["e", "f"]],
            recycle=True,
        )
        stats = await apply_virtual_scroll(page, _cfg(scroll_count=3))
        assert stats["captured"] == 6

    @pytest.mark.asyncio
    async def test_children_with_no_identity_signal_suppress_the_rewrite(self):
        """Nothing distinguishes them, so the merged set may be short — and a
        short set must not replace the live DOM."""

        class BlankPage(FakePage):
            def _child_records(self):
                return [{"key": "", "html": "<div></div>"} for _ in self._visible_children()]

        page = BlankPage(pages_of_children=[["a", "b"], ["c", "d"]], recycle=True)
        stats = await apply_virtual_scroll(page, _cfg(scroll_count=2))
        assert stats["unkeyable"] is True
        assert stats["merged"] is False
        assert page.rewritten is None

    @pytest.mark.asyncio
    async def test_duplicate_keys_within_one_batch_do_not_overwrite_each_other(self):
        """Two rows with identical text (or a duplicated id in generated markup)
        are still two rows."""

        class DupePage(FakePage):
            def _child_records(self):
                return [
                    {"key": "k:same", "html": f"<li>{c}</li>"} for c in self._visible_children()
                ]

        page = DupePage(pages_of_children=[["one", "two", "three"]], recycle=True)
        stats = await apply_virtual_scroll(page, _cfg(scroll_count=1))
        assert stats["captured"] == 3

    @pytest.mark.asyncio
    async def test_an_eval_failure_midway_keeps_what_was_already_captured(self):
        """A frame that detaches on a later step must not report captured: 0,
        which reads as "this container was always empty"."""

        class FlakyPage(FakePage):
            def __init__(self, **kw):
                super().__init__(**kw)
                self.calls = 0

            async def evaluate(self, script: str, arg=None):
                if "scrollHeight" in script:
                    self.calls += 1
                    if self.calls > 2:
                        raise RuntimeError("frame detached")
                return await super().evaluate(script, arg)

        page = FlakyPage(pages_of_children=[["<li>1</li>"], ["<li>2</li>"], ["<li>3</li>"]])
        stats = await apply_virtual_scroll(page, _cfg(scroll_count=5))
        assert stats["error"] == "page_eval_failed"
        assert stats["captured"] > 0
