"""The R reproduction of the headline statistics.

The fixture builds a phase4 folder the way the pipeline does — aligned returns and factors
in, Python's regressions / metrics / scores out — then hands the aligned data to r/verify.R
and asserts the two agree.  R recomputes the Newey-West sandwich in base R, so this is a
real second opinion on the alphas, t-statistics and scores, not a second call to statsmodels.
Skipped where Rscript or data.table is absent (Railway and cloud sessions have no R)."""
import shutil

import numpy as np
import pandas as pd
import pytest

from trackrecord import rverify as RV
from trackrecord import validation as V


def _phase4(tmp_path, n=120, seed=0):
    """A synthetic monthly manager written out in the phase4 layout."""
    rng = np.random.default_rng(seed)
    cells = pd.period_range("2015-01", periods=n, freq="M").astype(str)
    mkt = rng.normal(0.007, 0.042, n)
    df = pd.DataFrame(index=pd.Index(cells, name="cell"))
    df["MKT_RF"] = mkt
    for f, sd in [("SMB", 0.02), ("HML", 0.02), ("MOM", 0.025), ("RMW", 0.015), ("CMA", 0.015)]:
        df[f] = rng.normal(0.001, sd, n)
    df["RF"] = 0.0012
    df["r_p"] = 0.002 + 1.1 * df.MKT_RF + 0.3 * df.SMB - 0.2 * df.HML + rng.normal(0, 0.018, n) + df.RF
    df["r_b"] = df.MKT_RF + df.RF
    df["y"] = df.r_p - df.RF
    df["y_b"] = df.r_b - df.RF

    cfg = V.ValidationConfig(periods_per_year=12, factor_set="US", robustness_set="", headline_model="FF3")
    regs = V.factor_regressions(df, cfg, "US")
    mets = V.risk_metrics(df, 12)
    scrs = V.scores(df, regs, mets, pd.DataFrame(), cfg)
    p4 = tmp_path / "phase4"
    p4.mkdir()
    df.to_csv(p4 / "aligned_data.csv")
    regs.to_csv(p4 / "regressions.csv", index=False)
    mets.to_csv(p4 / "metrics.csv", index=False)
    scrs.to_csv(p4 / "scores.csv", index=False)
    return p4


def test_reports_unavailable_without_r(tmp_path):
    """No R, no crash — the caller gets a reason and the build skips the manager."""
    out = RV.verify_fund(tmp_path / "nothing-here")
    assert out["available"] is False and out["reason"]


@pytest.mark.skipif(shutil.which("Rscript") is None, reason="R not installed")
def test_r_reproduces_python_headline_numbers(tmp_path):
    p4 = _phase4(tmp_path)
    out = RV.verify_fund(p4)
    if not out["available"] and "data.table" in str(out.get("reason", "")):
        pytest.skip("data.table not installed")
    assert out["available"], out
    q = out["quantities"].set_index(["quantity", "model"])

    # every headline number the memo and the dashboards lead with is checked
    for model in ("CAPM", "FF3", "Carhart4", "FF5"):
        for quantity in ("alpha_annual", "se_annual", "t_alpha", "p_alpha", "r2"):
            assert (quantity, model) in q.index, (quantity, model)
    for quantity in ("annualized return", "annualized volatility", "Sharpe", "max drawdown", "alpha-maxing score"):
        assert (quantity, "-") in q.index, quantity

    assert out["max_abs_diff"] < 1e-8, out["quantities"].sort_values("abs_diff").tail()
    assert out["agree"]
    # the HAC lag is the statsmodels rule, not a default: floor(0.75 * n^(1/3))
    assert q.loc[("hac_lag", "FF3"), "r_value"] == int(np.floor(0.75 * 120 ** (1 / 3)))


@pytest.mark.skipif(shutil.which("Rscript") is None, reason="R not installed")
def test_build_writes_summary(tmp_path, monkeypatch):
    p4 = _phase4(tmp_path)
    funds = tmp_path / "out"
    (funds / "testfund").mkdir(parents=True)
    shutil.copytree(p4, funds / "testfund" / "phase4")
    monkeypatch.setattr(RV, "FUNDS_OUT", funds)
    man = RV.build(out_dir=tmp_path / "res", log=lambda *a: None)
    if not man["available"]:
        pytest.skip(man["r"])
    assert man["managers"] == 1 and man["disagreements"] == 0
    s = pd.read_csv(tmp_path / "res" / "summary.csv")
    assert s.slug.tolist() == ["testfund"] and bool(s.agree.iloc[0])
    assert abs(s.ff3_t_r.iloc[0] - s.ff3_t_python.iloc[0]) < 1e-8
    assert (tmp_path / "res" / "by_quantity.csv").exists()
