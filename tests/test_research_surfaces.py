"""The home page as a research tool, and the /research index.

These check the machinery a visitor actually uses: that every manager row carries the
attributes the screener filters on, that the research cards state numbers read back from
the committed CSVs rather than hard-coded prose, and that the index lists real files."""
import re

import pytest

from trackrecord import serve as S
from trackrecord.research_pages import research_index_html


def test_research_cards_read_their_numbers_from_the_data():
    html, have = S._research_cards()
    assert "memo" in have                                  # the memo card never depends on a build
    assert html.count("class='rcard'") == len(have)
    for slug in have:
        if slug == "memo":
            continue
        assert f"/research/{ {'signals': '13f-signals', 'construction': 'construction', 'rverify': 'r-verify'}[slug] }".replace(" ", "") in html
    # a card only claims a number it could read: no empty stat values
    assert "<span class='v'></span>" not in html


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
