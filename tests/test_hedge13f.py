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
    idx = pd.date_range("2020-01-31", "2021-06-30", freq="ME")
    a = pd.Series(100 * (1.01 ** np.arange(len(idx))), index=idx)      # +1%/month
    b = pd.Series(100 * (1.02 ** np.arange(len(idx))), index=idx)      # +2%/month
    c = pd.Series(np.nan, index=idx)                                   # never priced (delisted)
    flat = {f"F{i}": pd.Series(100.0, index=idx) for i in range(4)}   # filler names, 0% return
    return pd.DataFrame({"AAA": a, "BBB": b, "CCC": c, **flat})


def _fill(period, filed, w_each=0.0):
    """Four zero-return filler positions so the 5-name minimum is met without changing the math."""
    return [dict(period=period, filed=filed, cusip=f"F{i}", name=f"F{i}", cls="", value=0.0, weight=w_each) for i in range(4)]


def test_clone_rebalances_at_filing_month_and_drifts():
    h = pd.DataFrame([
        dict(period="2020-03-31", filed="2020-05-14", cusip="A", name="AAA", cls="", value=50.0, weight=0.5),
        dict(period="2020-03-31", filed="2020-05-14", cusip="B", name="BBB", cls="", value=50.0, weight=0.5),
        *_fill("2020-03-31", "2020-05-14"),
        dict(period="2020-06-30", filed="2020-08-14", cusip="B", name="BBB", cls="", value=100.0, weight=1.0),
        *_fill("2020-06-30", "2020-08-14"),
    ])
    cmap = {"A": "AAA", "B": "BBB", **{f"F{i}": f"F{i}" for i in range(4)}}
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
    assert list(summ.period) == ["2020-03-31", "2020-06-30"] and list(summ.n) == [6, 5]
    # no filing after August 2020 -> holdings go stale after STALE_MONTHS; nothing is held into 2021
    assert ret["2020-12-31"] == pytest.approx(0.02)
    assert ret["2021-01-31":].isna().all()


def test_unpriced_holding_drops_and_coverage_falls():
    h = pd.DataFrame([
        dict(period="2020-03-31", filed="2020-05-14", cusip="A", name="AAA", cls="", value=40.0, weight=0.4),
        dict(period="2020-03-31", filed="2020-05-14", cusip="C", name="CCC", cls="", value=60.0, weight=0.6),
        *_fill("2020-03-31", "2020-05-14"),
    ])
    ret, cov, summ = H.clone_returns(h, {"A": "AAA", "C": "CCC", **{f"F{i}": f"F{i}" for i in range(4)}}, _prices())
    # CCC has no prices -> 40% of value priced -> below the 60% floor -> NaN
    assert ret["2020-06-30":].isna().all()
    assert cov["2020-06-30"] == pytest.approx(0.4)
    assert summ.priced_share.iloc[0] == pytest.approx(0.4)


def test_too_few_priced_positions_is_nan():
    h = pd.DataFrame([dict(period="2020-03-31", filed="2020-05-14", cusip="A", name="AAA", cls="", value=100.0, weight=1.0)])
    ret, cov, summ = H.clone_returns(h, {"A": "AAA"}, _prices())
    assert ret.isna().all()            # one priced name is a bet, not a clone


def test_unmapped_cusip_is_ignored_and_dataset_insufficient(tmp_path):
    h = pd.DataFrame([dict(period="2020-03-31", filed="2020-05-14", cusip="A", name="AAA", cls="", value=100.0, weight=1.0), *_fill("2020-03-31", "2020-05-14")])
    ret, cov, summ = H.clone_returns(h, {"A": "AAA", **{f"F{i}": f"F{i}" for i in range(4)}}, _prices())
    fund = dict(slug="x", name="X", manager="m", style="ls", filers=[dict(cik="1")])
    meta = H.write_fund_dataset(fund, ret, cov, summ, tmp_path / "x")
    assert meta["status"] == "insufficient" and meta["months"] < 36


def test_longest_gap_free_stretch_is_kept(tmp_path):
    idx = pd.date_range("2015-01-31", "2024-12-31", freq="ME")
    ret = pd.Series(0.01, index=idx)
    ret.iloc[20:30] = np.nan          # a 10-month hole: 20 months before, 90 after
    cov = pd.Series(0.9, index=idx)
    summ = pd.DataFrame([dict(period="2015-03-31", rebalance=idx[0].date(), n=10, priced_share=0.9)])
    fund = dict(slug="y", name="Y", manager="m", style="ls", filers=[dict(cik="1")])
    meta = H.write_fund_dataset(fund, ret, cov, summ, tmp_path / "y")
    assert meta["status"] == "ok" and meta["months"] == 90 and meta["dropped_months"] == 20
    st = pd.read_csv(tmp_path / "y" / "statements.csv")
    assert st.period_end.min() == str(idx[30].date()) and len(st) == 90
