import numpy as np
import pytest

from trackrecord.reconcile import reconcile, Tolerance, ERROR
from helpers import ds, stmt, codes


def status(res, sid):
    return res.statements.set_index("statement_id").at[sid, "status"]


def test_clean_chain_is_verified():
    res = reconcile(ds([stmt("A_2001", 2001, 110.0, begin=100.0),
                        stmt("A_2002", 2002, 121.0, begin=110.0)]))
    assert status(res, "A_2002") == "verified"
    assert "CHAIN_BREAK" not in codes(res)
    assert res.summary()["errors"] == 0


def test_chain_break_flagged_with_numbers():
    res = reconcile(ds([stmt("A_2001", 2001, 110.0, begin=100.0),
                        stmt("A_2002", 2002, 121.0, begin=115.0)]))
    assert status(res, "A_2002") == "flagged"
    f = res.flags[res.flags.code == "CHAIN_BREAK"].iloc[0]
    assert f.severity == ERROR and "5.00" in f.detail


def test_first_statement_with_no_beginning_is_unverified():
    res = reconcile(ds([stmt("A_2001", 2001, 110.0)]))
    assert status(res, "A_2001") == "unverified"
    assert "INCEPTION" in codes(res, "A_2001")
    assert np.isnan(res.statements.dietz_return.iloc[0])


def test_inferred_beginning_alone_does_not_verify():
    # ending-only statements chained together: nothing independent confirms them
    res = reconcile(ds([stmt("A_2001", 2001, 100.0), stmt("A_2002", 2002, 110.0)]))
    assert status(res, "A_2002") == "unverified"
    assert res.statements.set_index("statement_id").at["A_2002", "beginning_source"] == "prior_ending"
    assert res.statements.set_index("statement_id").at["A_2002", "dietz_return"] == pytest.approx(0.10)


def test_positions_sum_verifies_and_mismatch_flags():
    pos = [dict(statement_id="A_2001", account_id="F_1", as_of_date="2001-12-31",
                identifier="AAPL", market_value=60.0),
           dict(statement_id="A_2001", account_id="F_1", as_of_date="2001-12-31",
                identifier="CASH", market_value=40.0)]
    ok = reconcile(ds([stmt("A_2001", 2001, 100.0)], positions=pos))
    assert status(ok, "A_2001") == "verified"
    pos[0]["market_value"] = 600.0
    bad = reconcile(ds([stmt("A_2001", 2001, 100.0)], positions=pos))
    assert status(bad, "A_2001") == "flagged"
    assert "POSITIONS_SUM_MISMATCH" in codes(bad, "A_2001")


def test_flows_and_pnl_checks():
    flows = [dict(account_id="F_1", date="2001-07-01", amount=50.0, flow_type="deposit"),
             dict(account_id="F_1", date="2001-09-01", amount=-10.0, flow_type="withdrawal")]
    s = stmt("A_2001", 2001, 150.0, begin=100.0, stated_deposits=50.0,
             stated_withdrawals=10.0, stated_pnl=10.0)
    res = reconcile(ds([s], flows=flows))
    assert status(res, "A_2001") == "verified"
    row = res.statements.iloc[0]
    assert row.net_flows == 40.0 and row.computed_pnl == pytest.approx(10.0)
    assert row.checks_possible == 3 and row.checks_passed == 3

    s["stated_pnl"] = 25.0   # statement disagrees
    res = reconcile(ds([s], flows=flows))
    assert "PNL_MISMATCH" in codes(res, "A_2001")

    res = reconcile(ds([s], flows=flows[:1]))   # withdrawal row never entered
    assert "FLOWS_MISMATCH" in codes(res, "A_2001")


def test_modified_dietz_known_value():
    # 365-day year, begin 1000, deposit 100 on Jul 2 (182 days to year end), end 1200
    # w = 182/365; r = (1200-1000-100) / (1000 + 100*182/365)
    flows = [dict(account_id="F_1", date="2001-07-02", amount=100.0, flow_type="deposit")]
    res = reconcile(ds([stmt("A_2001", 2001, 1200.0, begin=1000.0)], flows=flows))
    expected = 100 / (1000 + 100 * 182 / 365)
    assert res.statements.dietz_return.iloc[0] == pytest.approx(expected, abs=1e-12)


def test_stated_return_mismatch_is_warning_not_error():
    res = reconcile(ds([stmt("A_2001", 2001, 110.0, begin=100.0, stated_return_pct=15.0)]))
    assert "RETURN_MISMATCH" in codes(res, "A_2001")
    assert res.summary()["errors"] == 0
    ok = reconcile(ds([stmt("A_2001", 2001, 110.0, begin=100.0, stated_return_pct=10.4)]))
    assert "RETURN_MISMATCH" not in codes(ok, "A_2001")


def test_implausible_return_is_error():
    res = reconcile(ds([stmt("A_2001", 2001, 10000.0, begin=100.0)]))
    assert "IMPLAUSIBLE_RETURN" in codes(res, "A_2001")
    assert status(res, "A_2001") == "flagged"


def test_gap_orphan_and_no_beginning():
    flows = [dict(account_id="F_1", date="2002-06-01", amount=5.0, flow_type="deposit")]
    res = reconcile(ds([stmt("A_2001", 2001, 100.0, begin=90.0),
                        stmt("A_2003", 2003, 130.0)], flows=flows))
    assert "PERIOD_GAP" in codes(res, "A_2003")
    assert "NO_BEGINNING_VALUE" in codes(res, "A_2003")
    assert "ORPHAN_FLOW" in codes(res)
    assert res.flags[res.flags.code == "ORPHAN_FLOW"].statement_id.isna().all()


def test_gap_with_printed_beginning_skips_chain_but_keeps_return():
    res = reconcile(ds([stmt("A_2001", 2001, 100.0, begin=90.0),
                        stmt("A_2003", 2003, 130.0, begin=120.0)]))
    assert "CHAIN_SKIPPED" in codes(res, "A_2003")
    assert "CHAIN_BREAK" not in codes(res, "A_2003")
    assert res.statements.set_index("statement_id").at["A_2003", "dietz_return"] == pytest.approx(130 / 120 - 1)


def test_duplicate_and_overlap():
    dup = reconcile(ds([stmt("A_2001", 2001, 100.0), stmt("A_2001b", 2001, 100.0)]))
    assert "DUPLICATE_PERIOD" in codes(dup)
    s2 = stmt("A_2002", 2002, 110.0)
    s2["period_start"] = "2001-11-01"
    ovl = reconcile(ds([stmt("A_2001", 2001, 100.0), s2]))
    assert "PERIOD_OVERLAP" in codes(ovl, "A_2002")


def test_flow_on_period_boundaries_belongs_to_that_period():
    flows = [dict(account_id="F_1", date="2002-01-01", amount=1.0, flow_type="deposit"),
             dict(account_id="F_1", date="2002-12-31", amount=2.0, flow_type="deposit")]
    res = reconcile(ds([stmt("A_2001", 2001, 100.0), stmt("A_2002", 2002, 110.0)], flows=flows))
    st = res.statements.set_index("statement_id")
    assert st.at["A_2001", "n_flows"] == 0 and st.at["A_2002", "n_flows"] == 2
    assert "ORPHAN_FLOW" not in codes(res)


def test_accounts_are_independent():
    res = reconcile(ds([stmt("A_2001", 2001, 100.0, begin=90.0, acct="F_1"),
                        stmt("B_2002", 2002, 500.0, begin=400.0, acct="F_2")]))
    assert "CHAIN_BREAK" not in codes(res) and "PERIOD_GAP" not in codes(res)


def test_tolerance():
    t = Tolerance(abs_dollars=1.0, rel=1e-4)
    assert t.matches(100.0, 100.99)          # within $1
    assert not t.matches(100.0, 101.01)
    assert t.matches(1_000_000.0, 1_000_099.0)   # within 1bp
    assert not t.matches(1_000_000.0, 1_000_101.0)
    assert not t.matches(np.nan, 1.0)
