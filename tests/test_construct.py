"""The construction LP on a tiny universe: every constraint binds where it should, the alpha
ordering drives the tilts, names leaving the universe are sold, the trade list adds up, and —
when R with Rglpk is installed — the R twin lands on the same solution."""
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trackrecord import construct as C


def _case():
    alpha = pd.Series({"A": 2.0, "B": 1.0, "C": 0.0, "D": -1.0, "E": -2.0, "F": 0.5})
    bench = pd.Series({"A": 0.25, "B": 0.20, "C": 0.20, "D": 0.15, "E": 0.10, "F": 0.10})
    sectors = pd.Series({"A": "Tech", "B": "Tech", "C": "Health", "D": "Health", "E": "Energy", "F": "Energy"})
    return alpha, bench, sectors


def test_lp_respects_every_constraint_and_tilts_toward_alpha():
    alpha, bench, sectors = _case()
    c = C.Constraints(max_weight=0.30, active_band=0.05, sector_band=0.04, turnover=0.10, cost_bps=10)
    w, info = C.solve_lp(alpha, bench, sectors, bench.copy(), c)
    assert abs(w.sum() - 1) < 1e-9 and (w >= -1e-12).all()
    act = w - bench
    assert (act.abs() <= c.active_band + 1e-9).all() and (w <= c.max_weight + 1e-9).all()
    sec_act = act.groupby(sectors).sum()
    assert (sec_act.abs() <= c.sector_band + 1e-9).all()
    assert info["turnover"] <= c.turnover + 1e-9 and info["active_share"] <= c.active_share + 1e-9
    assert act["A"] > 0 and act["E"] < 0 and act["A"] >= act["B"] - 1e-9       # best alpha overweight, worst underweight
    assert info["expected_alpha"] > info["bench_alpha"]


def test_active_share_cap_binds():
    alpha, bench, sectors = _case()
    loose = C.Constraints(max_weight=0.30, active_band=0.10, sector_band=0.20, turnover=1.0, active_share=1.0)
    tight = C.Constraints(max_weight=0.30, active_band=0.10, sector_band=0.20, turnover=1.0, active_share=0.05)
    w1, i1 = C.solve_lp(alpha, bench, sectors, bench.copy(), loose)
    w2, i2 = C.solve_lp(alpha, bench, sectors, bench.copy(), tight)
    assert i2["active_share"] <= 0.05 + 1e-9 < i1["active_share"]
    assert i2["expected_alpha"] < i1["expected_alpha"]                    # the cap costs signal, as it should


def test_leaving_names_are_sold_and_turnover_relaxes_when_infeasible():
    alpha, bench, sectors = _case()
    w0 = pd.Series({"A": 0.2, "B": 0.2, "C": 0.2, "D": 0.1, "E": 0.1, "Z": 0.2})      # Z no longer in the universe
    c = C.Constraints(max_weight=0.30, active_band=0.05, sector_band=0.04, turnover=0.05, cost_bps=10)
    w, info = C.solve_lp(alpha, bench, sectors, w0, c)
    assert w.get("Z", 0.0) == 0.0
    assert info["relaxed"] >= 1 and info["turnover"] <= info["turnover_budget"] + 1e-9   # selling Z alone needs 20% > 5%


def test_trade_list_adds_up():
    alpha, bench, sectors = _case()
    c = C.Constraints(max_weight=0.30, active_band=0.05, sector_band=0.04, turnover=0.10)
    w, _ = C.solve_lp(alpha, bench, sectors, bench.copy(), c)
    px = pd.Series({t: 50.0 for t in alpha.index})
    tl = C.trade_list(w, bench, px, {}, sectors, alpha, bench, nav=1_000_000, cost_bps=10)
    assert abs(tl.trade_usd.sum()) < 1e-6                                   # fully invested before and after: net zero
    assert (tl.loc[tl.side == "BUY", "shares"] > 0).all() and (tl.loc[tl.side == "SELL", "shares"] < 0).all()
    assert abs(tl.est_cost_usd.sum() - tl.trade_usd.abs().sum() * 1e-3) < 1e-6


def test_ex_ante_te_is_zero_at_benchmark():
    rng = np.random.default_rng(0)
    rets = pd.DataFrame(rng.normal(0, 0.05, (36, 4)), columns=list("ABCD"))
    cov = C.shrunk_cov(rets)
    b = pd.Series(0.25, index=list("ABCD"))
    assert C.ex_ante_te(b, b, cov) < 1e-12
    assert C.ex_ante_te(pd.Series({"A": 0.5, "B": 0.5}), b, cov) > 0


@pytest.mark.skipif(shutil.which("Rscript") is None, reason="R not installed")
def test_r_twin_matches_python(tmp_path):
    alpha, bench, sectors = _case()
    c = C.Constraints(max_weight=0.30, active_band=0.05, sector_band=0.04, turnover=0.10, cost_bps=10)
    w0 = pd.Series({"A": 0.2, "B": 0.2, "C": 0.2, "D": 0.1, "E": 0.1, "Z": 0.2})
    w, info = C.solve_lp(alpha, bench, sectors, w0, c)
    pd.DataFrame({"ticker": alpha.index, "alpha": alpha.values, "bench": bench.values, "sector": sectors.values,
                  "w0": w0.reindex(alpha.index).fillna(0).values}).to_csv(tmp_path / "universe.csv", index=False)
    pd.DataFrame({"ticker": ["Z"], "w0": [0.2]}).to_csv(tmp_path / "leaving.csv", index=False)
    d = C.asdict(C.Constraints(max_weight=0.30, active_band=0.05, sector_band=0.04, turnover=info["turnover_budget"], cost_bps=10))
    pd.DataFrame({"key": list(d), "value": list(d.values())}).to_csv(tmp_path / "constraints.csv", index=False)
    pd.DataFrame({"ticker": w.index, "weight": w.values}).to_csv(tmp_path / "weights_python.csv", index=False)
    out = C.compare_with_r(tmp_path, log=lambda *a: None)
    if not out["available"] and "Rglpk" in str(out.get("reason", "")):
        pytest.skip("Rglpk not installed")
    assert out["available"], out
    assert abs(out["expected_alpha_r"] - out["expected_alpha_python"]) < 1e-6    # same optimum (weights may tie)
    assert out["max_abs_diff"] < 1e-6


def test_fund_of_funds_shrinkage_and_allocation():
    from trackrecord import fof as F
    M = pd.DataFrame({"alpha": [0.06, 0.02, -0.01, 0.00], "se": [0.05, 0.008, 0.02, 0.01], "resid_vol": [0.15, 0.04, 0.08, 0.05]}, index=list("ABCD"))
    Meb = F.shrink_alphas(M)
    assert (Meb.shrink_factor <= 1).all() and Meb.loc["B", "shrink_factor"] > Meb.loc["A", "shrink_factor"]   # precise estimate keeps more
    M2 = F.shrink_alphas(M, tau=0.02)
    assert abs(M2.loc["A", "alpha_shrunk"]) < abs(M.loc["A", "alpha"]) and M2.loc["B", "alpha_shrunk"] > 0.015
    idx = pd.date_range("2018-01-31", periods=60, freq="ME")
    rng = np.random.default_rng(0)
    U = pd.DataFrame(rng.normal(size=(60, 4)) * M.resid_vol.values / np.sqrt(12), index=idx, columns=list("ABCD"))
    S = F.residual_cov(U, list("ABCD"))
    w = F.allocate(M2.alpha_shrunk[M2.alpha_shrunk > 0], S, max_weight=0.6)
    assert abs(w.sum() - 1) < 1e-9 and (w <= 0.6 + 1e-9).all() and w.get("B", 0) > w.get("A", 0)               # the precise low-vol alpha dominates
