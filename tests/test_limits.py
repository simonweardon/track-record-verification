"""Due diligence on this work: the page that answers a reviewer's objections with numbers.

These tests check that the answers are computed rather than asserted — that the universe
comparison really widens the universe, that the reconciliation of the portfolio result with
its signal is arithmetic and not a claim, and that every figure printed on the page traces
back to a committed file."""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trackrecord import limits as L
from trackrecord.research_pages import limits_html

ROOT = Path(__file__).resolve().parents[1]
D = L.OUT_DIR
built = pytest.mark.skipif(not (D / "manifest.json").exists(), reason="limits not built")


@built
def test_every_published_file_exists_and_is_readable():
    man = json.loads((D / "manifest.json").read_text())
    assert man["managers"] > 50 and man["widest_universe"] > man["narrowest_universe"]
    for name in ("universe_ic", "coverage", "missing", "law", "costs", "capacity",
                 "reconstruction", "evidence", "cleaning", "verification", "freshness", "computation"):
        f = D / f"{name}.csv"
        assert f.exists(), name
        assert len(pd.read_csv(f)) > 0, name


@built
def test_the_universe_test_really_widens_the_universe():
    u = pd.read_csv(D / "universe_ic.csv")
    sizes = u.drop_duplicates("min_holders").set_index("min_holders").universe_avg
    assert sorted(sizes.index, reverse=True) == list(L.UNIVERSES)
    # fewer managers required to hold a name means strictly more names
    assert sizes.loc[1] > sizes.loc[2] > sizes.loc[5]
    assert sizes.loc[1] > 3 * sizes.loc[5]
    # the halves add up to the whole: each half's mean sits between the extremes of the pair
    for _, r in u.iterrows():
        assert min(r.ic_first, r.ic_second) <= r.ic + 1e-9
        assert r.ic <= max(r.ic_first, r.ic_second) + 1e-9


@built
def test_the_reconciliation_is_arithmetic():
    """The implied information ratio must be the product it says it is, and the standard error
    must be big enough that the page cannot claim the delivered result beats it."""
    law = pd.read_csv(D / "law.csv").set_index("key").value
    assert np.isclose(law.ir_implied, law.transfer * law.ic * np.sqrt(law.breadth), rtol=1e-9)
    assert law.ir_se > 0.1
    assert abs(law.ir_actual - law.ir_implied) < 3 * law.ir_se


@built
def test_costs_and_capacity_move_the_right_way():
    c = pd.read_csv(D / "costs.csv").sort_values("cost_bps")
    assert list(c.cost_bps) == sorted(L.COST_GRID)
    assert c.active_return.is_monotonic_decreasing and c.information_ratio.is_monotonic_decreasing
    cap = pd.read_csv(D / "capacity.csv").sort_values("nav")
    assert cap.cost_bps.is_monotonic_increasing and cap.max_days_of_volume.is_monotonic_increasing
    # the constraint that binds first is position size, not cost
    assert cap.cost_bps.max() < 100
    assert cap.trades_over_limit.max() > 0


@built
def test_a_manager_whose_disclosure_is_mostly_options_is_labelled_as_such():
    r = pd.read_csv(D / "reconstruction.csv")
    assert set(r.columns) >= {"slug", "confidence", "option_share", "put_share", "stock_share"}
    assert np.allclose(r.option_share + r.stock_share, 1.0)
    heavy = r[r.option_share >= L.OPTION_HEAVY]
    assert len(heavy) > 0
    assert (heavy.confidence == "Stock positions only").all()
    # a long/short manager is never sold as a close reconstruction
    assert not ((r.style == "ls") & (r.confidence == "Close")).any()


def test_confidence_downgrades_on_the_filing_not_the_label():
    assert L.confidence_for("conc", 0.0)[0] == "Close"
    assert L.confidence_for("conc", 0.5)[0] == "Stock positions only"
    assert L.confidence_for("ls", 0.0)[0] == "Long side only"
    assert L.confidence_for("multi", 0.9)[0] == "Not meaningful"


def test_years_needed_follow_the_square_rule():
    law = pd.DataFrame([dict(key="ir_actual", value=0.5)])
    ev = L.evidence_needed(law)
    assert np.allclose(ev.years_for_t2, (2.0 / ev.information_ratio) ** 2)
    assert ev.years_for_t2.is_monotonic_decreasing


@built
def test_the_page_states_the_uncomfortable_numbers():
    doc = limits_html()
    assert doc is not None
    man = json.loads((D / "manifest.json").read_text())
    text = re.sub(r"<[^>]+>", " ", doc)
    # the early years' coverage gap and the widest universe are both on the page
    assert f"{man['coverage_first']:.0%}" in text and f"{man['widest_universe']:,}" in text
    # the answer to "is the record long enough" is no, in the first word
    assert re.search(r"Is the record long enough to act on\?\s*</h2>\s*</div>\s*<div class=\"card\">\s*<p>No\.", doc)
    # the page names what the independent check cannot reach
    assert "not covered by any agreement between implementations" in doc
    # every file it points at is downloadable through the research route
    for rel in set(re.findall(r"/research/data/(limits/[a-z0-9_]+\.csv)", doc)):
        assert (ROOT / "data" / "research" / rel).exists(), rel


@built
def test_the_screener_shows_how_closely_each_record_tracks_the_fund():
    """The caveat has to live where the ranking is, not only on the method page."""
    from trackrecord import serve as S
    html = S.directory_html()
    assert 'id="f-rc"' in html
    rc = pd.read_csv(D / "reconstruction.csv").set_index("slug")
    rows = dict(re.findall(r"data-slug='([^']+)'[^>]*data-rc='([^']*)'", html))
    assert len(rows) > 50
    labelled = {k: v for k, v in rows.items() if v}
    assert len(labelled) > 50
    for slug, label in labelled.items():
        if slug in rc.index:
            assert label == str(rc.loc[slug, "confidence"]).lower(), slug
    # a hedged manager is never presented as a close reconstruction
    for slug in rc[rc["style"] == "ls"].index:
        assert rows.get(slug, "") != "close", slug
    # the export carries the column too
    assert "tracks_the_fund" in html
