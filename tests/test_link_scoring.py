"""Tests for relevance scoring of discovered links."""

from services.link_scoring import LinkScorer, tokenize


class TestTokenize:
    def test_lowercases_and_splits_on_non_alnum(self):
        assert tokenize("Pricing & Plans!") == ["pricing", "plans"]

    def test_drops_stopwords_and_single_chars(self):
        assert tokenize("the a x of our team") == ["team"]

    def test_empty_input(self):
        assert tokenize("") == []


class TestIntrinsicScoring:
    """No query supplied — scores come from link quality signals alone."""

    def setup_method(self):
        self.scorer = LinkScorer.for_query(None)

    def test_no_query_means_no_query_tokens(self):
        assert self.scorer.has_query is False

    def test_score_is_bounded(self):
        for url, text in [
            ("https://example.com/", "Home"),
            ("https://example.com/a/b/c/d/e/f/g", "x"),
            ("https://other.com/f.zip?a=1&b=2&utm_source=x", ""),
        ]:
            score = self.scorer.score(url=url, text=text, link_type="internal")
            assert 0.0 <= score <= 1.0

    def test_empty_url_scores_zero(self):
        assert self.scorer.score(url="", text="Pricing") == 0.0

    def test_descriptive_anchor_beats_generic_anchor(self):
        good = self.scorer.score(
            url="https://example.com/pricing", text="Pricing", link_type="internal"
        )
        generic = self.scorer.score(
            url="https://example.com/pricing", text="click here", link_type="internal"
        )
        assert good > generic

    def test_numeric_anchor_is_treated_as_low_value(self):
        pager = self.scorer.score(url="https://example.com/list", text="7", link_type="internal")
        labelled = self.scorer.score(
            url="https://example.com/list", text="Product list", link_type="internal"
        )
        assert labelled > pager

    def test_shallow_path_beats_deep_path(self):
        shallow = self.scorer.score(
            url="https://example.com/about", text="About us", link_type="internal"
        )
        deep = self.scorer.score(
            url="https://example.com/a/b/c/d/e/about", text="About us", link_type="internal"
        )
        assert shallow > deep

    def test_internal_beats_external_beats_unclassified_is_between(self):
        args = {"url": "https://example.com/careers", "text": "Careers"}
        internal = self.scorer.score(**args, link_type="internal")
        unknown = self.scorer.score(**args, link_type=None)
        external = self.scorer.score(**args, link_type="external")
        assert internal > unknown > external

    def test_nofollow_is_penalized(self):
        args = {"url": "https://example.com/blog", "text": "Blog", "link_type": "internal"}
        assert self.scorer.score(**args) > self.scorer.score(**args, rel="nofollow noopener")

    def test_tracking_params_are_penalized(self):
        clean = self.scorer.score(
            url="https://example.com/pricing", text="Pricing", link_type="internal"
        )
        tracked = self.scorer.score(
            url="https://example.com/pricing?utm_source=nl&utm_campaign=x&ref=y",
            text="Pricing",
            link_type="internal",
        )
        assert clean > tracked

    def test_non_html_target_is_penalized_but_not_zeroed(self):
        html = self.scorer.score(
            url="https://example.com/report", text="Annual report", link_type="internal"
        )
        pdf = self.scorer.score(
            url="https://example.com/report.pdf", text="Annual report", link_type="internal"
        )
        assert html > pdf > 0.0

    def test_html_extension_is_not_penalized(self):
        page = self.scorer.score(
            url="https://example.com/about.html", text="About", link_type="internal"
        )
        archive = self.scorer.score(
            url="https://example.com/about.zip", text="About", link_type="internal"
        )
        assert page > archive


class TestContextualScoring:
    def test_query_match_outranks_unrelated_link(self):
        scorer = LinkScorer.for_query("pricing plans")
        assert scorer.has_query is True
        relevant = scorer.score(
            url="https://example.com/pricing", text="Our plans", link_type="internal"
        )
        irrelevant = scorer.score(
            url="https://example.com/privacy", text="Privacy policy", link_type="internal"
        )
        assert relevant > irrelevant

    def test_slug_match_counts_more_than_anchor_text_match(self):
        scorer = LinkScorer.for_query("careers")
        in_slug = scorer.score(url="https://example.com/careers", text="Join", link_type="internal")
        in_text = scorer.score(url="https://example.com/x", text="Careers", link_type="internal")
        assert in_slug > in_text

    def test_prefix_match_handles_simple_morphology(self):
        scorer = LinkScorer.for_query("pricing")
        # "price" prefixes "pricing" at >= 4 chars, so it matches without a stemmer.
        matched = scorer.score(url="https://example.com/price", text="Cost", link_type="internal")
        unmatched = scorer.score(url="https://example.com/legal", text="Cost", link_type="internal")
        assert matched > unmatched

    def test_short_tokens_do_not_prefix_match_promiscuously(self):
        scorer = LinkScorer.for_query("api")
        # "api" is under the 4-char prefix floor, so it must not match "apiary-ish"
        # slugs by prefix alone — only an exact token match counts.
        exact = scorer.score(url="https://example.com/api", text="API", link_type="internal")
        loose = scorer.score(
            url="https://example.com/apiculture", text="Bees", link_type="internal"
        )
        assert exact > loose

    def test_underscores_and_encoding_in_slug_are_tokenized(self):
        scorer = LinkScorer.for_query("annual report")
        assert (
            scorer.score(
                url="https://example.com/annual_report%20final", text="", link_type="internal"
            )
            > 0.4
        )

    def test_query_of_only_stopwords_behaves_as_no_query(self):
        scorer = LinkScorer.for_query("the and of")
        assert scorer.has_query is False
        with_stopwords = scorer.score(
            url="https://example.com/pricing", text="Pricing", link_type="internal"
        )
        plain = LinkScorer.for_query(None).score(
            url="https://example.com/pricing", text="Pricing", link_type="internal"
        )
        assert with_stopwords == plain


class TestRankingBehaviour:
    """The end-to-end property that matters: the right link sorts to the top."""

    PAGE_LINKS = [
        {"url": "https://example.com/", "text": "Home", "type": "internal"},
        {"url": "https://example.com/blog/2024/01/a-post", "text": "A post", "type": "internal"},
        {"url": "https://example.com/pricing", "text": "Pricing", "type": "internal"},
        {"url": "https://twitter.com/example", "text": "Follow us", "type": "external"},
        {"url": "https://example.com/legal/privacy", "text": "Privacy", "type": "internal"},
        {"url": "https://example.com/x", "text": "click here", "type": "internal"},
        {"url": "https://example.com/media/deck.pdf", "text": "Deck", "type": "internal"},
    ]

    def _ranked(self, query):
        scorer = LinkScorer.for_query(query)
        scored = [
            (link["url"], scorer.score(url=link["url"], text=link["text"], link_type=link["type"]))
            for link in self.PAGE_LINKS
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return [url for url, _ in scored]

    def test_query_puts_the_matching_page_first(self):
        assert self._ranked("pricing")[0] == "https://example.com/pricing"

    def test_social_and_junk_links_sink_without_a_query(self):
        ranked = self._ranked(None)
        assert ranked.index("https://example.com/pricing") < ranked.index(
            "https://twitter.com/example"
        )
        assert ranked.index("https://example.com/pricing") < ranked.index("https://example.com/x")
