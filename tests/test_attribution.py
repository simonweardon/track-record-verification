import numpy as np
import pandas as pd
import pytest

from trackrecord.synthetic import generate
from trackrecord.reconcile import reconcile
from trackrecord.returns import period_returns
from trackrecord.composite import build as build2
from trackrecord import attribution as A


@pytest.fixture(scope="module")
def ref():
    from trackrecord.reference import load_all
    try:
        r = load_all()
    except Exception as e:
        pytest.skip(f"reference data unavailable: {e}")
    return r


@pytest.fixture(scope="module")
def stack(ref):
    d, man = generate(seed=7, defects=True)
    if man.market_source != "DEV_MKT":
        pytest.skip("synthetic not built from reference market")
    res = reconcile(d)
    pr = period_returns(res.statements, d.flows)
    r2 = build2(res.statements, d.flows, pr, d.accounts, ref)
    r3 = A.build(r2.periods, d.positions, r2.composite, ref)
    return d, r2, r3, man


def test_policy_weights_are_plausible(ref):
    pw = A.infer_policy_weights(ref, range(1997, 2026))
    assert 0.35 < pw.loc[1997, "US"] < 0.55
    assert 0.55 < pw.loc[2025, "US"] < 0.75
    assert ((pw.US + pw.INTL).round(9) == 1).all()


def test_beta_decomposition_sums_and_recovers_beta(stack):
    _, _, r3, man = stack
    bd = r3.beta_decomp
    total = bd.rf_contribution + bd.beta_contribution + bd.alpha_plus_residual
    assert np.allclose(total, bd.portfolio)
    assert bd.attrs["beta"] == pytest.approx(man.true_beta, abs=0.1)
    assert bd.attrs["alpha"] == pytest.approx(man.true_alpha, abs=0.012)


def test_allocation_identity_holds_every_year(stack):
    _, _, r3, _ = stack
    al = r3.allocation[r3.allocation.available == True]
    lhs = al.allocation_effect + al.selection_timing_residual + al.policy_construction_error
    assert np.allclose(lhs, al.total_active)
    assert np.allclose(al.total_active, al.portfolio - al.benchmark_actual)
    assert (al.coverage > 0.8).all()


def test_sleeve_weights_sum_to_one_and_alpha_lands_in_residual(stack):
    _, _, r3, man = stack
    al = r3.allocation[r3.allocation.available == True]
    w = al[["w_US", "w_INTL", "w_CASH", "w_UNCLASSIFIED"]].sum(axis=1)
    assert np.allclose(w, 1.0)
    # synthetic weights are random, so allocation effect should be ~0 and the
    # injected alpha should sit in the residual
    assert abs(al.allocation_effect.mean()) < 0.01
    assert al.selection_timing_residual.mean() == pytest.approx(man.true_alpha, abs=0.012)


def test_timing_tests_report_no_timing_on_synthetic(stack):
    _, _, r3, _ = stack
    tm = r3.timing.set_index("model")
    assert set(tm.index) == {"Treynor-Mazuy", "Henriksson-Merton"}
    assert (tm.gamma_p > 0.05).all()      # no timing was injected


def test_selection_level_with_hand_made_prices(stack, ref):
    d, r2, _, _ = stack
    # give every synthetic holding exactly its sleeve benchmark return -> selection effect 0
    years = list(r2.composite.index)
    rows = []
    po = d.positions
    for ident, ac in po[["identifier", "asset_class"]].drop_duplicates().itertuples(index=False):
        col = {"us_equity": "US_MKT", "intl_equity": "DXUS_MKT"}.get(ac)
        if col is None:
            continue
        for y in years:
            rows.append(dict(identifier=ident, year=y, total_return=A.annual(ref, col, [y])[y]))
    prices = pd.DataFrame(rows)
    sel = A.selection_attribution(r2.periods, po, prices, r2.composite, ref)
    s = sel[sel.available == True]
    assert len(s) > 20
    assert np.allclose(s.selection_effect, 0.0, atol=1e-9)
    assert np.allclose(s.buy_and_hold_return + s.trading_timing_residual, s.index.map(r2.composite.gross))


def test_write_phase3(tmp_path, stack):
    _, _, r3, _ = stack
    p = A.write_phase3(r3, tmp_path)
    t = p.read_text()
    assert "Level 1" in t and "Level 2" in t and "Level 3" in t and "Not run" in t
