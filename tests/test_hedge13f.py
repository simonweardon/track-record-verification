"""Clone mechanics on a hand-made universe: rebalance at end of filing month,
drift within the quarter, unpriced names drop and renormalize, coverage recorded."""
import numpy as np
import pandas as pd
import pytest

from trackrecord import hedge13f as H


def test_name_normalization():
    assert H._norm("APPLE INC") == "APPLE"
    assert H._norm("Alphabet Inc.") == "ALPHABET"
    assert H._norm("BERKSHIRE HATHAWAY INC DEL CL B") == "BERKSHIRE HATHAWAY"
    assert H._norm("Apple Hospitality REIT, Inc.") == "APPLE HOSPITALITY REIT"


def _prices():
    idx = pd.date_range("2020-01-31", "2020-12-31", freq="ME")
    a = pd.Series(100 * (1.01 ** np.arange(12)), index=idx)      # +1%/month
    b = pd.Series(100 * (1.02 ** np.arange(12)), index=idx)      # +2%/month
    c = pd.Series(np.nan, index=idx)                             # never priced (delisted)
    return pd.DataFrame({"AAA": a, "BBB": b, "CCC": c})


def test_clone_rebalances_at_filing_month_and_drifts():
    h = pd.DataFrame([
        dict(period="2020-03-31", filed="2020-05-14", cusip="A", name="AAA", cls="", value=50.0, weight=0.5),
        dict(period="2020-03-31", filed="2020-05-14", cusip="B", name="BBB", cls="", value=50.0, weight=0.5),
        dict(period="2020-06-30", filed="2020-08-14", cusip="B", name="BBB", cls="", value=100.0, weight=1.0),
    ])
    cmap = {"A": "AAA", "B": "BBB"}
    ret, cov, summ = H.clone_returns(h, cmap, _prices())
    # nothing before the first rebalance (end of May), first return in June
    assert ret[:"2020-05-31"].isna().all()
    assert ret["2020-06-30"] == pytest.approx(0.5 * 0.01 + 0.5 * 0.02)
    # July: drifted weights -> slightly more in BBB than 50/50
    wa, wb = 0.5 * 1.01, 0.5 * 1.02
    assert ret["2020-07-31"] == pytest.approx((wa * 0.01 + wb * 0.02) / (wa + wb))
    # after the August filing (rebalance end of Aug), 100% BBB from September
    assert ret["2020-09-30"] == pytest.approx(0.02)
    assert np.allclose(cov.dropna(), 1.0)
    assert list(summ.period) == ["2020-03-31", "2020-06-30"] and list(summ.n) == [2, 1]


def test_unpriced_holding_drops_and_coverage_falls():
    h = pd.DataFrame([
        dict(period="2020-03-31", filed="2020-05-14", cusip="A", name="AAA", cls="", value=40.0, weight=0.4),
        dict(period="2020-03-31", filed="2020-05-14", cusip="C", name="CCC", cls="", value=60.0, weight=0.6),
    ])
    ret, cov, summ = H.clone_returns(h, {"A": "AAA", "C": "CCC"}, _prices())
    # CCC has no prices -> only AAA is investable, but coverage reflects 40% of value priced -> below 50% -> NaN
    assert ret["2020-06-30":].isna().all()
    assert cov["2020-06-30"] == pytest.approx(0.4)
    assert summ.priced_share.iloc[0] == pytest.approx(0.4)


def test_unmapped_cusip_is_ignored_and_dataset_insufficient(tmp_path):
    h = pd.DataFrame([dict(period="2020-03-31", filed="2020-05-14", cusip="A", name="AAA", cls="", value=100.0, weight=1.0)])
    ret, cov, summ = H.clone_returns(h, {"A": "AAA"}, _prices())
    fund = dict(slug="x", name="X", manager="m", style="ls", filers=[dict(cik="1")])
    meta = H.write_fund_dataset(fund, ret, cov, summ, tmp_path / "x")
    assert meta["status"] == "insufficient" and meta["months"] < 36
