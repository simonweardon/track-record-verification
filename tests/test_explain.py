"""Every computed number on the site carries a "How & why" note.

A tile without a note is a failure, not a gap: the registry in trackrecord/explain.py must
cover every label on every rendered page (research notes, memo, simulation, dashboard) and
every stat on the home and area cards."""
from pathlib import Path

import pytest

from trackrecord import explain as E
from trackrecord import research_pages as R
from trackrecord.areas import tool_cards, home_html
from trackrecord import serve as S

ROOT = Path(__file__).resolve().parents[1]


def test_annotate_adds_one_note_per_tile_and_is_idempotent():
    doc = ('<div class="tiles"><div class="tile"><div class="tl">Sharpe</div><div class="tv">1.2</div>'
           '<div class="td muted">market 0.8</div></div><div class="tile"><div class="tl">Unknown thing</div>'
           '<div class="tv">3</div></div></div>')
    out = E.annotate(doc, "/f/x/memo")
    assert out.count('<details class="how hw">') == 1
    assert out.index("How &amp; why") > out.index("market 0.8")          # appended inside the tile, after its detail line
    assert E.annotate(out, "/f/x/memo") == out
    # a tile that already carries a method note is left alone
    keep = '<div class="tile"><div class="tl">Sharpe</div><div class="tv">1</div><details class="how"><summary>x</summary></details></div>'
    assert E.annotate(keep, "/f/x/memo") == keep


PAGES = [("/research/alpha-lab", R.alphalab_html), ("/research/risk-model", R.riskmodel_html), ("/research/construction", R.construction_html),
         ("/research/r-verify", R.rverify_html), ("/research/decay", R.decay_html), ("/research/fund-of-funds", R.fof_html),
         ("/research/13f-signals", R.signals13f_html), ("/research/limits", R.limits_html)]


@pytest.mark.parametrize("path,fn", PAGES, ids=[p for p, _ in PAGES])
def test_every_research_tile_has_a_note(path, fn):
    doc = fn()
    if doc is None:
        pytest.skip(f"{path} not built")
    labels = E.tile_labels(doc)
    missing = [l for l in labels if E.lookup(l, path) is None]
    assert not missing, missing
    out = E.annotate(doc, path)
    assert out.count("how hw") == len(labels)


def test_every_card_stat_has_a_note():
    from trackrecord.areas import method_card
    cards = [c for cs in tool_cards().values() for c in cs] + [c for c in (method_card(),) if c]
    for c in cards:
        for _, label in c["stats"]:
            assert E.lookup(label, c["href"]), (c["href"], label)
    doc = home_html(S.page, 1, 1)
    assert doc.count("how hw") == sum(len(c["stats"]) for c in cards)
    assert "<a class='rcard'" not in doc and "<div class='rcard'" in doc      # notes cannot live inside a link


def test_memo_and_dashboard_tiles_have_notes():
    out = ROOT / "output" / "funds"
    built = sorted(d for d in out.glob("*") if (d / "dashboard.html").exists()) if out.exists() else []
    if not built:
        pytest.skip("no built manager (output/ is gitignored)")
    from trackrecord.memo import build_memo
    d = built[0]
    memo = build_memo(d, ROOT / "data" / "funds" / d.name)
    labels = E.tile_labels(memo)
    assert labels and all(E.lookup(l, f"/f/{d.name}/memo") for l in labels), [l for l in labels if not E.lookup(l, "memo")]
    dash = E.annotate((d / "dashboard.html").read_text(), "dashboard")   # idempotent: the build already did this
    for t in E.tiles(dash):
        assert "<details" in t, t[:200]


def test_simulation_tile_labels_are_covered():
    # the labels simpages.py renders (a simulation is not built in a fresh clone, so they are listed here)
    for lab in ["Alpha-maxing score", "Wealth-management score", "FF3 alpha", "Sharpe · IR", "Gross /yr", "Net of manager fees /yr", "Max drawdown", "Best / worst month"]:
        assert E.lookup(lab, "/sim/abc123/"), lab


def test_charts_carry_a_how_this_was_calculated_note():
    """Figures (not just tiles) on the three Manager Analysis notes, and on Active."""
    needed = {
        "/research/13f-signals": (R.signals13f_html, "This is not a made-up path", 6),
        "/research/decay": (R.decay_html, "This is not a growth-of-a-dollar chart", 3),
        "/research/fund-of-funds": (R.fof_html, "N on the axis is a count of managers", 4),
    }
    for path, (fn, marker, n) in needed.items():
        doc = fn()
        if doc is None:
            pytest.skip(f"{path} not built")
        assert marker in doc, (path, marker)
        assert doc.count('class="how fig"') >= n, (path, doc.count('class="how fig"'))
        assert "How this was calculated" in doc
        assert "Where the numbers come from." in doc


def test_active_story_tiles_have_notes():
    from trackrecord.areas import area_html
    doc = area_html(S.page, "active")
    labels = E.tile_labels(doc)
    assert labels, "the story should carry the same numbered tiles as the four notes"
    missing = [l for l in labels if E.lookup(l, "/active") is None]
    assert not missing, missing
    out = E.annotate(doc, "/active")
    assert out.count("how hw") == len(labels)
