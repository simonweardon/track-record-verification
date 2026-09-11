import numpy as np
import pandas as pd
import pytest

from trackrecord.synthetic import generate
from trackrecord.reconcile import reconcile
from trackrecord.returns import period_returns
from trackrecord.composite import build as build2
from trackrecord import validation as V


@pytest.fixture(scope="module")
def refs():
    from trackrecord.reference import load_all
    try:
        return load_all(), load_all(freq="A")
    except Exception as e:
        pytest.skip(f"reference data unavailable: {e}")


@pytest.fixture(scope="module")
def r4(refs):
    ref, refA = refs
    d, man = generate(seed=7, defects=True)
    if man.market_source != "DEV_MKT":
        pytest.skip("synthetic not built from reference market")
    res = reconcile(d)
    pr = period_returns(res.statements, d.flows)
    r2 = build2(res.statements, d.flows, pr, d.accounts, ref)
    return V.build(r2.composite, refA, V.ValidationConfig(n_boot=1000, n_cohort=2000)), man


def test_capm_ci_contains_true_alpha_and_beta_near_one(r4):
    res, man = r4
    R = res.regressions.set_index(["factor_set", "model"])
    c = R.loc[("DEV", "CAPM")]
    assert c.ci_low_annual <= man.true_alpha <= c.ci_high_annual
    assert c.b_MKT_RF == pytest.approx(man.true_beta, abs=0.1)
    assert c.r2 > 0.9
    assert c.n == 29


def test_all_models_both_sets_ran(r4):
    res, _ = r4
    R = res.regressions
    assert set(R.model) == set(V.MODELS) and set(R.factor_set) == {"DEV", "US"}
    assert R.alpha_annual.notna().all()
    assert (R.ci_low_annual < R.alpha_annual).all() and (R.alpha_annual < R.ci_high_annual).all()


def test_bootstrap_and_cohort_agree_with_parametric(r4):
    res, _ = r4
    R = res.regressions.set_index(["factor_set", "model"]).loc[("DEV", "CAPM")]
    b = res.bootstrap.set_index("model").loc["CAPM"]
    assert abs(b.p_null_one_sided - R.p / 2) < 0.06        # same question, similar answer
    assert b.alpha_ci_low_annual < R.alpha_annual < b.alpha_ci_high_annual
    c = res.cohort.set_index("variant").loc["same market path"]
    assert c.beta == pytest.approx(R.b_MKT_RF, abs=1e-9)
    assert 80 <= c.percentile_by_headline_excess <= 100
    assert abs((100 - c.percentile_by_t_alpha) / 100 - b.p_null_one_sided) < 0.08


def test_cohort_median_zero_skill_manager_has_near_zero_excess(r4):
    res, _ = r4
    for _, c in res.cohort.iterrows():
        assert abs(c.cohort_excess_p50) < 0.01


def test_metrics_shape(r4):
    res, _ = r4
    m = res.metrics.set_index("metric")
    assert m.loc["beta vs benchmark", "portfolio"] == pytest.approx(1.04, abs=0.1)
    assert -0.5 < m.loc["max drawdown (cell-level; understated on annual cells)", "portfolio"] < 0
    assert m.loc["periods", "portfolio"] == 29


def test_rolling_and_subperiods(r4):
    res, _ = r4
    r = res.rolling
    assert r.excess_5.notna().sum() == 25 and r.alpha_10.notna().sum() == 20
    ok = r.dropna(subset=["alpha_10"]); assert (ok.alpha_10_ci_low <= ok.alpha_10).all() and (ok.alpha_10 <= ok.alpha_10_ci_high).all()
    sp = res.subperiods.set_index(["model", "segment"])
    assert sp.loc[("CAPM", "before 2010"), "n"] + sp.loc[("CAPM", "2010 onward"), "n"] == 29
    d = sp.loc[("CAPM", "difference (post − pre)"), "alpha_annual"]
    assert d == pytest.approx(sp.loc[("CAPM", "2010 onward"), "alpha_annual"] - sp.loc[("CAPM", "before 2010"), "alpha_annual"])


def test_max_drawdown_known():
    r = pd.Series([0.10, -0.20, -0.25, 0.50])
    assert V.max_drawdown(r) == pytest.approx(0.8 * 0.75 - 1)


def test_write_phase4(tmp_path, r4):
    res, _ = r4
    t = V.write_phase4(res, tmp_path).read_text()
    assert "one-paragraph answer" in t and "Noise floor" in t and "| CAPM |" in t


def test_tail_risk_helpers_known_values():
    r = pd.Series([-0.10, -0.05, 0.0, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08])
    assert V.var_hist(r) == pytest.approx(np.percentile(r, 5))
    assert V.var_param(r) == pytest.approx(r.mean() - 1.6448536 * r.std(ddof=1), abs=1e-6)
    assert V.es_hist(r) == pytest.approx(-0.10)          # only the worst value sits at/below the 5th pct
    assert np.isnan(V.var_hist(pd.Series([0.1, 0.2])))


def test_risk_by_year_monthly_and_annual():
    idx = pd.period_range("2001-01", "2003-12", freq="M")
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"r_p": rng.normal(0.01, 0.05, len(idx)), "r_b": rng.normal(0.008, 0.04, len(idx))}, index=idx)
    m = V.risk_by_year(df, 12)
    assert list(m.index) == [2001, 2002, 2003] and (m.n == 12).all()
    y1 = df.loc["2001-01":"2001-12"]
    assert m.loc[2001, "return_p"] == pytest.approx((1 + y1.r_p).prod() - 1)
    assert m.loc[2001, "vol_p"] == pytest.approx(y1.r_p.std(ddof=1) * np.sqrt(12))
    assert m.loc[2001, "variance_p"] == pytest.approx(y1.r_p.var(ddof=1) * 12)
    assert m.loc[2001, "var95_hist_p"] == pytest.approx(np.percentile(y1.r_p, 5))
    assert m.loc[2001, "worst_p"] == y1.r_p.min()
    assert (m.var95_hist_p <= m.var95_hist_p.abs()).all()   # VaR reported as a (negative) return level
    assert np.isnan(m.loc[2001, "trailing_vol_p"]) and not np.isnan(m.loc[2003, "trailing_vol_p"])
    assert m.attrs["within_year_observable"]

    a = pd.DataFrame({"r_p": rng.normal(0.08, 0.15, 15), "r_b": rng.normal(0.07, 0.14, 15)}, index=range(2000, 2015))
    ay = V.risk_by_year(a, 1)
    assert ay.vol_p.isna().all() and not ay.attrs["within_year_observable"]
    assert ay.trailing_vol_p.notna().sum() >= 10
    assert ay.loc[2014, "trailing_var95_hist_p"] == pytest.approx(np.percentile(a.r_p.tail(10), 5))


def test_risk_return_points(r4):
    res, _ = r4
    rr = res.risk_return.set_index("kind")
    assert {"composite", "benchmark", "modelnet", "rf"} <= set(rr.index)
    assert rr.loc["rf", "vol"] == 0 and np.isnan(rr.loc["rf", "sharpe"])
    assert (rr.loc["account"].n >= 3).all() and len(rr.loc["account"]) >= 8
    c = rr.loc["composite"]
    assert c.sharpe == pytest.approx((c.ret - rr.loc["rf", "ret"]) / c.vol)
    assert 0.15 < c.vol < 0.25
