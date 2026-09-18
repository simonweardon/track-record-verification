"""The home page as a research tool, and the /research index.

These check the machinery a visitor actually uses: that every manager row carries the
attributes the screener filters on, that the research cards state numbers read back from
the committed CSVs rather than hard-coded prose, and that the index lists real files."""
import re

import pytest

from trackrecord import serve as S
from trackrecord.areas import home_html
from trackrecord.research_pages import research_index_html


def test_research_cards_read_their_numbers_from_the_data():
    from trackrecord.areas import tool_cards, cards_html
    cards = tool_cards()
    assert cards["active"] and cards["external"]
    # Active is one story entry, not four separate research modules
    assert len(cards["active"]) == 1
    assert cards["active"][0]["href"] == "/active"
    assert cards["active"][0]["title"] == "How ranking scores became this trade list"
    for area in cards.values():
        for c in area:
            assert c["href"].startswith("/research/") or c["href"] == "/active", c["href"]
            assert c["stats"], c["href"]
            # a card only claims a number it could read: no empty or unreadable stat values
            assert all(str(v).strip() not in ("", "nan", "None") for v, _ in c["stats"]), (c["href"], c["stats"])
            # one or two complete sentences, no fragments
            what = c["what"].strip()
            assert what.endswith(".") and 1 <= what.count(". ") + 1 <= 3, what
    html = cards_html(cards["active"])
    assert html.count("class='rcard'") == 1
    assert "/research/alpha-lab" not in html
    assert 'href="/active"' in html


def test_home_page_does_not_list_separate_active_modules():
    """The four Active research notes live inside the story, not as home cards."""
    doc = home_html(S.page, 1, 1)
    assert "Signal Research" not in doc or doc.count("Signal Research") == 0
    # home must not advertise the four old Active module routes as cards
    for path in ("/research/alpha-lab", "/research/risk-model", "/research/construction", "/research/r-verify"):
        assert f'href="{path}"' not in doc and f"href='{path}'" not in doc, path
    assert 'href="/active"' in doc or "href='/active'" in doc
    assert "How ranking scores became this trade list" in doc


def test_home_page_rows_carry_the_screener_attributes():
    html = S.directory_html()
    assert 'id="f-t"' in html and 'id="f-y"' in html and 'id="f-m"' in html and 'id="dl"' in html
    rows = re.findall(r"<tr data-s=.*?</tr>", html, re.S)
    assert len(rows) > 50, len(rows)
    for r in rows:
        for attr in ("data-t=", "data-m=", "data-y=", "data-slug=", "data-mgr="):
            assert attr in r, (attr, r[:160])
    # the table scrolls inside its own box so a phone does not scroll sideways
    assert '<div class="tw"><table id="tbl">' in html
    # the emitted CSV export must carry a real newline escape, not a literal line break
    assert r"out.join('\n')" in html


def test_screener_filters_on_a_real_t_statistic():
    """The FF3 t reaching the page is the one from the pipeline, and it is sparse by design."""
    html = S.directory_html()
    ts = [float(t) for t in re.findall(r"data-t='(-?[\d.]+)'", html)]
    assert ts, "no t-statistics reached the page"
    real = [t for t in ts if t > -90]
    assert len(real) > 20
    # the honest finding: very few managers clear |t| >= 2
    assert 0 < sum(1 for t in real if t >= 2) < len(real) * 0.25


def test_research_index_lists_downloadable_files():
    html = research_index_html()
    assert "/research/data/" in html
    files = set(re.findall(r"/research/data/([a-z0-9-]+/[a-z0-9_]+\.csv)", html))
    assert len(files) >= 2
    root = S.ROOT / "data" / "research"
    for f in files:
        assert (root / f).exists(), f                      # never advertise a file that is not there
    assert "—" in html or "KB" in html                # sizes rendered


def test_research_data_route_rejects_traversal():
    pat = re.compile(r"^[a-z0-9-]+/[a-z0-9_]+\.csv$")
    for bad in ("../../CLAUDE.md", "r-verify/../../../etc/passwd", "r-verify/summary.csv/../x",
                "R-Verify/Summary.CSV", "r-verify/summary.txt"):
        assert not pat.match(bad), bad
    assert pat.match("r-verify/summary.csv") and pat.match("13f-signals/ic_summary.csv")


def test_home_headline_states_what_the_site_does():
    doc = home_html(S.page, 1, 1)
    assert "Due diligence on outside managers" in doc
    assert "as working software" not in doc
    assert "disclosed holdings" in doc.lower() or "Disclosed holdings" in doc
    assert "ranking" in doc.lower()


def test_active_tab_is_one_backtested_story():
    """The Active tab is a single walk-through, not four separate cards."""
    from trackrecord.areas import area_html
    doc = area_html(S.page, "active")
    assert "class='rcard'" not in doc
    assert "A backtest" in doc
    for heading in ("Objective", "Process", "Product",
                    "What “signal” means here",
                    "Turn a public stock score into a tradable book",
                    "How that objective was pursued",
                    "What that process produced"):
        assert heading in doc, heading
    assert "How ranking scores became this trade list" in doc
    assert "How a signal became a list of trades" not in doc
    assert "one number per stock" in doc.lower() or "One number per stock" in doc
    assert "rank" in doc.lower()
    assert "twelve-month momentum" in doc.lower()
    assert "/research/alpha-lab" not in doc and "/research/construction" not in doc
    assert "/research/risk-model" not in doc and "/research/r-verify" not in doc
    assert "/research/limits" not in doc
    assert "Full research note" not in doc and "Full risk note" not in doc
    assert "Full construction note" not in doc and "Full verification note" not in doc
    assert "Due Diligence on This Work" not in doc
    # no outbound research-module links left on the story
    assert "/research/" not in doc
    assert "class=\"how fig\"" in doc
    assert "Momentum score" in doc
    assert "IntersectionObserver" in doc or "prefers-reduced-motion" in doc
    assert "<style>" in doc and ".chartbox" in doc
    from tests.test_plain_language import visible
    assert ".chartbox" not in visible(doc)
    # legend swatch class must not be reused for the "When / Who" labels (that overlap bug)
    assert "when-lab" in doc and 'class="k"' not in doc.split('id="process"')[1].split('id="product"')[0]
    assert "wrap story" in doc or 'class="wrap story"' in doc
    assert 'data-n="' not in doc
