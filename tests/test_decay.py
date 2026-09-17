"""Manager decay model mechanics: filing features come out as defined (turnover, new / sold shares,
the 13F unit step), the walk-forward never trains on a label window that has not closed, the AUC is
the rank formula, the evaluation recovers a planted signal and stays at 0.5 on noise, and the R twin
rebuilds the same panel (skipped without Rscript / data.table)."""
import shutil

import numpy as np
import pandas as pd
import pytest

from trackrecord import decay as D


def _books():
    """Two quarters of one manager: B sold, C bought, the rest held; every value ×1000 in Q2 (the 13F unit step)."""
    rows = []
    q1, q2 = "2020-03-31", "2020-06-30"
    for cusip, v1, v2 in [("A", 40.0, 40.0), ("B", 30.0, None), ("C", None, 30.0), ("D", 20.0, 20.0), ("E", 10.0, 10.0), ("F", 5.0, 5.0)]:
        if v1 is not None: rows.append(dict(slug="m", period=q1, cusip=cusip, value=v1, shares=v1))
        if v2 is not None: rows.append(dict(slug="m", period=q2, cusip=cusip, value=v2 * 1000, shares=v2))
    g = pd.DataFrame(rows)
    g["filed"] = g.period; g["name"] = g.cusip; g["ticker"] = g.cusip
    g["w_all"] = g.value / g.groupby(["slug", "period"]).value.transform("sum")
    g = g.sort_values(["slug", "period", "value"], ascending=[True, True, False])
    g["rank"] = g.groupby(["slug", "period"]).cumcount() + 1
    g["n_book"] = g.groupby(["slug", "period"]).cusip.transform("size")
    g["formation"] = pd.to_datetime(g.period).map(lambda p: (p.to_period("M") + 2).to_timestamp("M"))
    g["prev_period"] = (pd.to_datetime(g.period).dt.to_period("Q") - 1).dt.end_time.dt.strftime("%Y-%m-%d")
    have = set(zip(g.slug, g.period))
    g["has_prev"] = [(s, p) in have for s, p in zip(g.slug, g.prev_period)]
    key = g.set_index(["slug", "period", "cusip"]).shares
    g["shares_prev"] = key.reindex(pd.MultiIndex.from_arrays([g.slug, g.prev_period, g.cusip])).values
    g["new"] = g.has_prev & g.shares_prev.isna()
    return g.reset_index(drop=True)


def test_filing_features_match_their_definitions():
    f = D.filing_features(_books()).set_index("period")
    q2 = f.loc["2020-06-30"]
    assert abs(q2.turnover - 30 / 105) < 1e-9        # B (30 of 105) left, C (30) arrived: overlap 75 / 105
    assert abs(q2.new_frac - 1 / 5) < 1e-9           # C is one of five names
    assert abs(q2.sold_frac - 1 / 5) < 1e-9          # B was one of five names last quarter
    assert abs(q2.book_growth) < 1e-9                # the ×1000 unit step is not growth
    w = np.array([40, 30, 20, 10, 5]) / 105
    assert abs(q2.top1_w - w[0]) < 1e-9 and abs(q2.hhi - (w ** 2).sum()) < 1e-9 and abs(q2.top10_w - 1.0) < 1e-9
    assert abs(q2.n_positions - np.log(5)) < 1e-9 and abs(q2.crowding - 1.0) < 1e-9
    q1 = f.loc["2020-03-31"]
    assert pd.isna(q1.turnover) and pd.isna(q1.new_frac) and pd.isna(q1.sold_frac) and pd.isna(q1.book_growth)


def test_auc_is_the_rank_formula():
    assert D.auc([1, 0, 1, 0], [0.9, 0.1, 0.8, 0.2]) == 1.0
    assert D.auc([1, 0, 1, 0], [0.1, 0.9, 0.2, 0.8]) == 0.0
    assert abs(D.auc([1, 0, 1, 0], [0.5, 0.5, 0.5, 0.5]) - 0.5) < 1e-12
    assert np.isnan(D.auc([1, 1], [0.2, 0.3]))


def _panel(signal: float, seed: int = 0, managers: int = 60, dates: int = 40):
    rng = np.random.default_rng(seed)
    F = pd.date_range("2014-08-31", periods=dates, freq="3ME")
    rows = []
    for k, d in enumerate(F):
        regime = rng.normal(scale=0.10)                                   # a year that hits everyone
        for m in range(managers):
            x = rng.normal(size=len(D.FEATURES))
            fwd = regime + signal * x[0] + rng.normal(scale=0.08)
            rows.append(dict(slug=f"m{m}", period=str(d.date()), formation=d, filed=str(d.date()), fwd_excess=fwd, fwd_return=fwd, **dict(zip(D.FEATURES, x))))
    P = pd.DataFrame(rows)
    P["y"] = (P.fwd_excess < 0).astype(float)
    P.loc[P.formation > F[-5], "y"] = np.nan                               # the last year has no label yet
    return P


def test_walk_forward_respects_the_embargo_and_finds_a_planted_signal():
    P, last = D.walk_forward(_panel(signal=0.06), log=lambda *a: None)
    tested = P[P.p_xgboost.notna()]
    first = tested.formation.min()
    # by the first prediction, MIN_TRAIN_DATES label windows must have closed: 12 dates + 4 quarters of horizon
    assert first >= P.formation.min() + pd.DateOffset(months=3 * D.MIN_TRAIN_DATES + D.HORIZON - 3)
    assert P.loc[P.formation < first, "p_logistic"].isna().all()
    S, byd = D.evaluate(P)
    S = S.set_index("model")
    assert S.loc["logistic", "auc_cs_mean"] > 0.6 and S.loc["logistic", "auc_cs_t"] > 3
    assert S.loc["xgboost", "auc_cs_mean"] > 0.58
    assert S.loc["logistic", "spread_mean"] < 0                             # riskiest fifth did worse
    assert abs(S.loc["persist", "auc_cs_mean"] - 0.5) < 0.08               # excess_12 is noise here
    assert last["coef"].abs().idxmax() == D.FEATURES[0]


def test_noise_scores_as_a_coin_flip_within_dates_despite_regimes():
    P, _ = D.walk_forward(_panel(signal=0.0, seed=3), log=lambda *a: None)
    S, _ = D.evaluate(P)
    S = S.set_index("model")
    for m in ("logistic", "xgboost"):
        assert abs(S.loc[m, "auc_cs_mean"] - 0.5) < 0.05 and abs(S.loc[m, "auc_cs_t"]) < 2.5
    U = D.univariate(P)
    assert (U.ic_t.abs() < 3).all()


@pytest.mark.skipif(shutil.which("Rscript") is None, reason="R not installed")
def test_r_twin_rebuilds_the_same_panel(tmp_path):
    from trackrecord.decay import OUT_DIR
    if not (OUT_DIR / "panel.csv").exists() or not (OUT_DIR / "factors.csv").exists():
        pytest.skip("decay outputs not built")
    # the R script writes panel_r.csv beside its inputs, so run it on a copy: a machine without
    # R's xgboost would otherwise quietly overwrite the committed comparison with a thinner one
    for f in ("panel.csv", "factors.csv"):
        shutil.copy(OUT_DIR / f, tmp_path / f)
    out = D.compare_with_r(tmp_path, log=lambda *a: None)
    if not out["available"] and "data.table" in str(out.get("reason", "")):
        pytest.skip("data.table not installed")
    assert out["available"], out
    assert out["rows_matched"] == out["rows_python"] == out["rows_r"]
    assert out["max_abs_diff"] < 1e-6
    for c, v in out["by_column"].items():
        assert v["same_missing"] == 1.0, c
