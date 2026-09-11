import numpy as np
import pandas as pd
import pytest

from trackrecord.synthetic import generate
from trackrecord.reconcile import reconcile
from trackrecord.returns import period_returns
from trackrecord.composite import build, CompositeConfig, classify, series_stats
from helpers import ds, stmt


@pytest.fixture(scope="module")
def ref():
    from trackrecord.reference import load_all
    try:
        r = load_all()
    except Exception as e:
        pytest.skip(f"reference data unavailable: {e}")
    if "DEV_MKT" not in r or r.loc["1996-01-31":"2025-12-31", "DEV_MKT"].isna().any():
        pytest.skip("reference data incomplete")
    return r


@pytest.fixture(scope="module")
def synth(ref):
    d, man = generate(seed=7, defects=True)
    if man.market_source != "DEV_MKT":
        pytest.skip("synthetic data not built from reference market")
    res = reconcile(d)
    pr = period_returns(res.statements, d.flows)
    return build(res.statements, d.flows, pr, d.accounts, ref), man


def test_composite_recovers_injected_alpha(synth):
    r2, man = synth
    c = r2.composite
    arith_excess = (c.gross - c.benchmark).mean()
    assert arith_excess == pytest.approx(man.true_alpha, abs=0.006)   # per-account noise + beta≠1
    assert r2.composite.n.min() >= 2


def test_closed_accounts_included_for_full_periods_only(synth):
    r2, _ = synth
    p = r2.periods
    fide = p[(p.account_id == "FIDE_1011") & p.eligible].cell.tolist()
    assert fide == [2005, 2006, 2007, 2008]           # opened Mar-2004, closed Mar-2009
    reasons = p[p.account_id == "FIDE_1011"].set_index("cell").exclusion_reason
    assert reasons[2004] == "partial_period" and reasons[2009] == "partial_period"
    assert (r2.accounts.set_index("account_id").loc["FIDE_1011", "status"]) == "closed/terminated"


def test_survivors_only_uses_fewer_accounts_when_closed_accounts_were_live(synth):
    r2, _ = synth
    c = r2.composite
    assert (c.loc[2005:2008, "n_survivors"] < c.loc[2005:2008, "n"]).all()
    assert (c.loc[2020:2025, "n_survivors"] == c.loc[2020:2025, "n"]).all()
    assert not np.isclose(series_stats(c.survivors_only, c.years)["annualized"],
                          series_stats(c.gross, c.years)["annualized"], atol=1e-6)


def test_non_discretionary_excluded_entirely(synth):
    r2, _ = synth
    p = r2.periods[r2.periods.account_id == "FIDE_1008"]
    assert not p.eligible.any()
    assert (p.exclusion_reason == "non_discretionary").all()


def test_flagged_excluded_by_default_but_includable(synth, ref):
    r2, _ = synth
    p = r2.periods.set_index("statement_id")
    assert p.loc["SCHW_1001_2003", "exclusion_reason"] == "flagged"
    assert r2.composite.loc[2003, "n_excluded_flagged"] >= 1
    d, _ = generate(seed=7, defects=True)
    res = reconcile(d)
    pr = period_returns(res.statements, d.flows)
    r2b = build(res.statements, d.flows, pr, d.accounts, ref, CompositeConfig(include_flagged=True))
    assert r2b.periods.set_index("statement_id").loc["SCHW_1001_2003", "eligible"]


def test_benchmark_equals_direct_compounding_when_all_share_one(synth, ref):
    r2, _ = synth
    y = 2010
    direct = float((1 + ref.loc[f"{y}-01-31":f"{y}-12-31", "DEV_MKT"]).prod() - 1)
    assert r2.composite.loc[y, "benchmark"] == pytest.approx(direct, abs=1e-12)


def test_model_net_never_above_gross(synth):
    r2, _ = synth
    c = r2.composite
    assert (c.model_net <= c.gross + 1e-12).all()


def test_pooled_irr_present_and_sane(synth):
    r2, _ = synth
    pooled = r2.irr[r2.irr.account_id == "POOLED_DISCRETIONARY"].irr.iloc[0]
    g = series_stats(r2.composite.gross, r2.composite.years)["annualized"]
    assert abs(pooled - g) < 0.05


def test_aggregate_equals_asset_weighted_without_flows():
    sts = [stmt("A_2001", 2001, 110.0, begin=100.0, acct="A"),
           stmt("B_2001", 2001, 360.0, begin=300.0, acct="B")]
    d = ds(sts)
    res = reconcile(d)
    pr = period_returns(res.statements, d.flows)
    r2 = build(res.statements, d.flows, pr, None, None)
    c = r2.composite.loc[2001]
    assert c.gross == pytest.approx((10 + 60) / 400)
    assert c.asset_weighted == pytest.approx(c.gross)
    assert c.equal_weighted == pytest.approx((0.10 + 0.20) / 2)
    assert "No accounts.csv" in r2.notes[0]


def test_aggregate_with_flows_is_pooled_dietz():
    flows = [dict(account_id="A", date="2001-07-02", amount=100.0, flow_type="deposit")]
    sts = [stmt("A_2001", 2001, 220.0, begin=100.0, acct="A"),
           stmt("B_2001", 2001, 330.0, begin=300.0, acct="B")]
    d = ds(sts, flows=flows)
    res = reconcile(d)
    pr = period_returns(res.statements, d.flows)
    r2 = build(res.statements, d.flows, pr, None, None)
    w = (pd.Timestamp("2001-12-31") - pd.Timestamp("2001-07-02")).days / 365
    expected = (550 - 400 - 100) / (400 + 100 * w)
    assert r2.composite.loc[2001, "gross"] == pytest.approx(expected)


def test_dispersion_only_with_enough_accounts(synth):
    r2, _ = synth
    c = r2.composite
    assert c.loc[c.n < 6, "dispersion_sd"].isna().all()
    assert c.loc[c.n >= 6, "dispersion_sd"].notna().all()


def test_series_stats_never_chains_across_a_hole():
    from trackrecord.composite import series_stats, longest_run
    r = pd.Series([0.1, 0.1, np.nan, 0.5, 0.5, 0.5], index=[2001, 2002, 2003, 2004, 2005, 2006])
    y = pd.Series(1.0, index=r.index)
    s = series_stats(r, y)
    assert (s["first"], s["last"], s["periods"]) == (2004, 2006, 3)
    assert s["truncated"] is True
    assert s["annualized"] == pytest.approx(0.5)
    # non-adjacent index labels are a hole too
    r2 = pd.Series([0.1, 0.1, 0.1], index=[2001, 2002, 2005])
    assert longest_run(r2).index.tolist() == [2001, 2002]
