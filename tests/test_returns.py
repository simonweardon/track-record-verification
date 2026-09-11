import numpy as np
import pandas as pd
import pytest

from trackrecord.returns import (xirr, model_net, FeeSchedule, chain, annualize,
                                 period_returns, contiguous_segments, segment_cashflows)
from trackrecord.reconcile import reconcile
from helpers import ds, stmt


def test_xirr_simple_doubling_in_two_years():
    r = xirr(pd.Series(["2000-01-01", "2002-01-01"]), pd.Series([-100.0, 200.0]))
    assert r == pytest.approx(2 ** (365.25 / 731) - 1, abs=1e-6)   # 731 days incl. leap day


def test_xirr_with_intermediate_deposit_matches_known():
    # -100 at t0, -100 at t=1y, +231 at t=2y: 100(1+r)^2 + 100(1+r) = 231 -> r = 0.10
    r = xirr(pd.Series(["2000-01-01", "2001-01-01", "2002-01-01"]),
             pd.Series([-100.0, -100.0, 231.0]))
    assert r == pytest.approx(0.10, abs=1e-3)


def test_xirr_no_sign_change_is_nan():
    assert np.isnan(xirr(pd.Series(["2000-01-01", "2001-01-01"]), pd.Series([-1.0, -1.0])))


def test_chain_and_annualize():
    assert chain(pd.Series([0.10, -0.05, 0.20])) == pytest.approx(1.1 * 0.95 * 1.2 - 1)
    assert annualize(1.0, 2.0) == pytest.approx(np.sqrt(2) - 1)
    assert np.isnan(annualize(np.nan, 2.0))


def test_model_net_management_only():
    g = pd.Series([0.10, 0.10])
    n = model_net(g, pd.Series([1.0, 1.0]), FeeSchedule(mgmt_pct=0.01, perf_pct=0.0))
    assert n.tolist() == pytest.approx([1.10 * 0.99 - 1] * 2)


def test_model_net_performance_fee_with_high_water_mark():
    fee = FeeSchedule(mgmt_pct=0.0, perf_pct=0.20, high_water_mark=True)
    # +20% -> fee on 0.20 -> net +16%; then -10% -> no fee; then +10%: nav 1.16*0.9*1.1=1.1484 < hwm 1.16 -> no fee
    n = model_net(pd.Series([0.20, -0.10, 0.10]), pd.Series([1.0] * 3), fee)
    assert n.iloc[0] == pytest.approx(0.16)
    assert n.iloc[1] == pytest.approx(-0.10)
    assert n.iloc[2] == pytest.approx(0.10)
    # then +5%: nav 1.1484*1.05=1.20582 > hwm 1.16 -> fee 0.2*(0.04582) -> nav 1.196656
    n = model_net(pd.Series([0.20, -0.10, 0.10, 0.05]), pd.Series([1.0] * 4), fee)
    assert n.iloc[3] == pytest.approx(1.196656 / 1.1484 - 1, abs=1e-6)


def test_default_schedule_is_documented():
    assert "0.50% management + 20% of gains above high-water mark" == FeeSchedule().describe()


def _pr(statements, flows=()):
    d = ds(statements, flows=flows)
    res = reconcile(d)
    return period_returns(res.statements, d.flows), res


def test_inception_period_is_measured_from_first_flow_and_not_full():
    flows = [dict(account_id="F_1", date="2001-07-02", amount=1000.0, flow_type="deposit")]
    pr, _ = _pr([stmt("A_2001", 2001, 1050.0, begin=0.0)], flows)
    row = pr.iloc[0]
    assert row.eff_start == pd.Timestamp("2001-07-02")
    assert not row.full_period
    # 183-day window; w for a day-one flow = 182/183 -> r = 50 / (1000*182/183)
    assert row.r == pytest.approx(50 / (1000 * 182 / 183))


def test_closing_period_ends_at_final_withdrawal():
    flows = [dict(account_id="F_1", date="2002-04-01", amount=-1100.0, flow_type="withdrawal")]
    pr, _ = _pr([stmt("A_2001", 2001, 1000.0, begin=900.0), stmt("A_2002", 2002, 0.0, begin=1000.0)], flows)
    row = pr.set_index("statement_id").loc["A_2002"]
    assert row.eff_end == pd.Timestamp("2002-04-01")
    assert not row.full_period
    assert row.r == pytest.approx(0.10)     # 1000 -> 1100 taken out on the last effective day (w=0)


def test_full_period_flag_and_segments_split_on_gap():
    pr, _ = _pr([stmt("A_2001", 2001, 110.0, begin=100.0),
                 stmt("A_2002", 2002, 121.0, begin=110.0),
                 stmt("A_2004", 2004, 150.0, begin=140.0)])
    assert pr.full_period.tolist() == [True, True, True]
    segs = contiguous_segments(pr)
    assert [len(s) for s in segs] == [2, 1]
    assert segment_cashflows(None, ds([]).flows) is None


def test_account_cashflows_investor_view():
    flows = [dict(account_id="F_1", date="2001-06-01", amount=50.0, flow_type="deposit"),
             dict(account_id="F_1", date="2002-06-01", amount=-20.0, flow_type="withdrawal")]
    d = ds([stmt("A_2001", 2001, 160.0, begin=100.0), stmt("A_2002", 2002, 150.0, begin=160.0)], flows=flows)
    res = reconcile(d)
    pr = period_returns(res.statements, d.flows)
    cf = segment_cashflows(contiguous_segments(pr)[0], d.flows)
    assert cf.amount.tolist() == [-100.0, -50.0, 20.0, 150.0]
    assert cf.date.iloc[0] == pd.Timestamp("2001-01-01") and cf.date.iloc[-1] == pd.Timestamp("2002-12-31")


def test_flagged_period_breaks_the_chain():
    pr, _ = _pr([stmt("A_2001", 2001, 110.0, begin=100.0),
                 stmt("A_2002", 2002, 12100.0, begin=110.0),      # x100 typo -> IMPLAUSIBLE_RETURN
                 stmt("A_2003", 2003, 130.0, begin=121.0),         # chain break vs 12100
                 stmt("A_2004", 2004, 143.0, begin=130.0)])
    segs = contiguous_segments(pr)
    assert [s.statement_id.tolist() for s in segs] == [["A_2001"], ["A_2004"]]
    assert len(contiguous_segments(pr, break_on_flagged=False)) == 1
