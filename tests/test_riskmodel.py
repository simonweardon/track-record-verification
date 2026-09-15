"""Risk model mechanics: the cross-sectional fit recovers planted factor returns under the industry
constraint, EWMA weights sum to one, and a portfolio's variance splits into factor and specific parts."""
import numpy as np
import pandas as pd

from trackrecord import riskmodel as RM


def _cross_section(seed=0, n=600):
    rng = np.random.default_rng(seed)
    styles = ["momentum", "value"]
    secs = ["Tech", "Health", "Energy"]
    g = pd.DataFrame({"ticker": [f"T{i}" for i in range(n)], "momentum": rng.normal(size=n), "value": rng.normal(size=n),
                      "sector": rng.choice(secs, n, p=[0.5, 0.3, 0.2])})
    f_ind = {"Tech": 0.01, "Health": -0.01}
    counts = g.sector.value_counts()
    f_ind["Energy"] = -(counts["Tech"] * 0.01 + counts["Health"] * -0.01) / counts["Energy"]     # count-weighted sum = 0
    g["fwd"] = 0.005 + 0.02 * g.momentum - 0.01 * g.value + g.sector.map(f_ind) + rng.normal(scale=0.001, size=n)
    return g, styles, secs, f_ind


def test_fit_month_recovers_factor_returns_and_constraint():
    g, styles, secs, f_ind = _cross_section()
    f, u, r2 = RM.fit_month(g, styles, secs)
    assert abs(f["market"] - 0.005) < 2e-3 and abs(f["momentum"] - 0.02) < 1e-3 and abs(f["value"] + 0.01) < 1e-3
    for s, v in f_ind.items():
        assert abs(f[f"ind:{s}"] - v) < 2e-3
    counts = g.sector.value_counts()
    assert abs(sum(counts[s] * f[f"ind:{s}"] for s in secs)) < 1e-9          # neutrality constraint holds exactly
    assert r2 > 0.95 and len(u) == len(g)


def test_ewma_weights_and_decomposition():
    w = RM._ewma_weights(60, 24)
    assert abs(w.sum() - 1) < 1e-12 and w[-1] > w[0]
    F = pd.DataFrame(np.diag([0.0004, 0.0001]), index=["market", "momentum"], columns=["market", "momentum"])
    X = pd.DataFrame({"market": [1.0, 1.0], "momentum": [1.0, -1.0]}, index=["A", "B"])
    D = pd.Series([0.0009, 0.0016], index=["A", "B"])
    d = RM.decompose(pd.Series({"A": 0.5, "B": 0.5}), X, F, D, ["momentum"])
    # momentum exposure nets to zero; factor var = 1² × 0.0004; specific = 0.25×0.0009 + 0.25×0.0016
    assert abs(d["factor"] - np.sqrt(0.0004 * 12)) < 1e-9
    assert abs(d["specific"] - np.sqrt((0.25 * 0.0009 + 0.25 * 0.0016) * 12)) < 1e-9
    assert abs(d["exposures"]["momentum"]) < 1e-12 and d["n_names"] == 2
