"""The rules that keep a phone from scrolling sideways.

None of this is about how the site looks; it is about the three things that actually
broke at 390px and would break again silently. A chart drawn at 860 user units and
scaled to a third of that takes its axis labels down to four pixels, so every chart is
wrapped in a .chartbox that scrolls instead. A grid or flex item whose min-content is
wider than the screen drags the whole page with it, so the phone media query zeroes
those minimums. And the simulate page lifts rules out of the dashboard stylesheet by
line, which must not pull a line out of a media block and apply it at every width.
"""
import re

from trackrecord import dashboard as D
from trackrecord import serve as S
from trackrecord.simpages import _dash_rules

MOBILE = "@media (max-width: 720px)"


def _mobile_block(css: str, marker: str = MOBILE) -> str:
    """The text of the phone media query, brace-matched."""
    start = css.index(marker)
    depth, i = 0, css.index("{", start)
    for j in range(i, len(css)):
        if css[j] == "{":
            depth += 1
        elif css[j] == "}":
            depth -= 1
            if depth == 0:
                return css[start:j + 1]
    raise AssertionError("unbalanced media query")


def test_every_chart_is_wrapped_in_a_scrollable_box():
    cells = ["2020-01", "2020-02", "2020-03", "2020-04"]
    series = [dict(name="Composite", values=[1.0, 1.1, 1.05, 1.2], cls="s1")]
    drawings = [
        D.line_chart(cells, series, uid="t1"),
        D.diverging_bars(["a", "b", "c"], [0.1, -0.2, 0.05]),
    ]
    for svg in drawings:
        assert svg.startswith('<div class="chartbox"><svg'), svg[:60]
        assert svg.count("</svg></div>") == 1
        # the box is what scrolls, and the chart keeps a legible width inside it
        assert svg.count("<div") == svg.count("</div>")


def test_a_long_value_cannot_stretch_its_grid_track():
    # not phone-only: one nowrap value in a tile pushed a 844px tablet sideways too
    rule = D.CSS[D.CSS.index(".wrap > *, section > *"):]
    rule = rule[:rule.index("}") + 1]
    for cls in (".tiles > *", ".tile > *", ".card > *", ".two > *", ".grid2 > *"):
        assert cls in rule, cls
    assert "min-width: 0" in rule
    assert "white-space: nowrap" not in D.CSS[D.CSS.index(".tv.small"):D.CSS.index(".tv.small") + 80]


def test_the_phone_rules_collapse_the_multi_column_tracks():
    block = _mobile_block(D.CSS)
    for cls in (".two", ".grid2", ".tiles", ".verdict .top"):
        assert cls in block, cls
    # multi-column tracks with a 300-420px minimum can never fit a 390px screen
    assert "grid-template-columns: 1fr" in block
    assert ".chartbox { overflow-x: auto" in block
    # the toolbar is sticky at top: 0; a second sticky element there is hidden behind it
    assert ".banner { position: static" in block


def test_the_toolbar_is_one_scrolling_row_on_a_phone():
    block = _mobile_block(S.NAV_CSS, "@media(max-width:720px)")
    assert "flex-wrap:nowrap" in block and "overflow-x:auto" in block
    assert "min-height:40px" in block
    # and the tool you are on is scrolled into view rather than left off the edge
    assert "scrollLeft" in S.toolbar("/external")


def test_form_controls_are_16px_so_ios_does_not_zoom_the_page_in():
    shell = _mobile_block(S.page("t", "<p>x</p>"), "@media(max-width:720px)")
    assert "font-size:16px" in shell
    screener = S.directory_html()
    assert ".filters select{width:100%;font-size:16px" in screener
    assert "#tbl{min-width:620px}" in screener      # the table scrolls; rows stay compact


def test_simulate_only_lifts_top_level_rules_from_the_dashboard_sheet():
    rules = _dash_rules()
    assert ".chart" in rules and ".tile" in rules          # it still gets what it came for
    assert "@media" not in rules
    for line in rules.splitlines():
        # a line pulled out of a media block would apply at every width
        assert not re.match(r"\s*(grid-template-columns|font-size):", line), line
    assert rules.count("{") == rules.count("}")
