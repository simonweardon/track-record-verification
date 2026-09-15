"""A fundamental factor risk model, Barra-style, on the research universe.

Each month, stock returns are regressed cross-sectionally on the stocks' exposures:

    r_it = f_mkt,t + Σ_s X_is,t f_s,t + Σ_j D_ij f_j,t + u_it

    X   style exposures — the alpha lab's point-in-time z-scores (momentum, reversal,
        volatility, size, value, profitability, cash flow, earnings yield), so the risk
        model and the alpha model describe the same stock the same way
    D   industry dummies (Yahoo sectors); the industry factor returns are constrained to
        sum to zero (count-weighted) so that the market factor is identified
    u   the stock-specific return

From the time series of factor returns f_t and residuals u_it:
    F   factor covariance — exponentially weighted (half-life 24 months) over the trailing
        window, so the model adapts without whipsawing
    Δ   specific variance per stock — EWMA of squared residuals, shrunk toward the
        cross-sectional median for stocks with short histories

For any portfolio w (or active weights w − b):
    σ² = x'Fx + w'Δw,   x = X'w
    reported as total, factor and specific risk, and by factor group — which is what a
    Barra Aegis / BPM report shows for a portfolio.

Calibration is tested the way risk models are tested: the bias statistic.  For each month,
random long-only portfolios and the benchmark get a predicted volatility; the standard
deviation of realised return ÷ predicted volatility should be ~1 (above 1 = the model
under-forecasts risk).  Outputs: data/research/risk-model/.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .alphalab import build_panel, SIGNALS
from .construct import load_sectors

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "research" / "risk-model"
STYLES = ["momentum", "reversal", "low_vol", "size", "value", "profitability", "cash_flow", "earnings_yield"]
HALF_LIFE = 24
WINDOW = 60
MIN_HIST = 12


def _ewma_weights(n: int, half_life: int) -> np.ndarray:
    w = 0.5 ** (np.arange(n)[::-1] / half_life)
    return w / w.sum()


def fit_month(g: pd.DataFrame, styles: list[str], sectors: list[str]) -> tuple[pd.Series, pd.Series, float]:
    """One cross-sectional regression with the industry-neutrality constraint.  Returns factor
    returns (market, styles, every industry), residuals by ticker, R²."""
    y = g.fwd.values
    Xs = g[styles].fillna(0.0).values
    ind = g.sector.values
    counts = pd.Series(ind).value_counts()
    secs = [s for s in sectors if s in counts.index]
    if len(secs) < 2:
        return pd.Series(dtype=float), pd.Series(dtype=float), np.nan
    last = secs[-1]
    D = np.zeros((len(g), len(secs) - 1))
    for j, s in enumerate(secs[:-1]):
        D[:, j] = (ind == s).astype(float) - (counts[s] / counts[last]) * (ind == last).astype(float)     # Σ n_s f_s = 0
    X = np.column_stack([np.ones(len(g)), Xs, D])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    r2 = 1 - resid.var() / y.var() if y.var() > 0 else np.nan
    f = dict(market=beta[0]); f.update({s: beta[1 + i] for i, s in enumerate(styles)})
    fi = {s: beta[1 + len(styles) + j] for j, s in enumerate(secs[:-1])}
    fi[last] = -sum(counts[s] * fi[s] for s in secs[:-1]) / counts[last]
    f.update({f"ind:{s}": v for s, v in fi.items()})
    return pd.Series(f), pd.Series(resid, index=g.ticker.values), float(r2)


def build_model(log=print) -> dict:
    panel, info = build_panel(log)
    sectors, _ = load_sectors()
    panel["sector"] = panel.ticker.map(sectors).fillna("Unknown")
    styles = [s for s in STYLES if s in panel and panel[s].notna().mean() > 0.3]
    secs = sorted(panel.sector.unique())
    F_rows, R2, resid = {}, {}, {}
    for m, g in panel.dropna(subset=["fwd"]).groupby("month"):
        f, u, r2 = fit_month(g, styles, secs)
        if len(f):
            F_rows[m] = f; R2[m] = r2; resid[m] = u
    fr = pd.DataFrame(F_rows).T.sort_index()            # month (the month whose return is explained: fwd of the signal month)
    fr.index = [pd.Timestamp(m) for m in fr.index]
    U = pd.DataFrame(resid).T.sort_index(); U.index = [pd.Timestamp(m) for m in U.index]
    log(f"risk model: {len(fr)} months, {len(styles)} styles + {len(secs)} industries, avg R² {np.nanmean(list(R2.values())):.2f}")
    return dict(panel=panel, factor_returns=fr, residuals=U, r2=pd.Series(R2), styles=styles, sectors=secs, info=info)


def factor_cov(fr: pd.DataFrame, asof: pd.Timestamp, window: int = WINDOW, half_life: int = HALF_LIFE) -> pd.DataFrame:
    """Row m of fr explains the return m → m+1, which is not known at m: use rows strictly before asof."""
    h = fr[fr.index < asof].tail(window).fillna(0.0)
    w = _ewma_weights(len(h), half_life)
    mu = (h.values * w[:, None]).sum(axis=0)
    X = h.values - mu
    C = (X * w[:, None]).T @ X
    return pd.DataFrame(C, index=h.columns, columns=h.columns)


def specific_var(U: pd.DataFrame, asof: pd.Timestamp, window: int = WINDOW, half_life: int = HALF_LIFE) -> pd.Series:
    h = U[U.index < asof].tail(window)
    w = _ewma_weights(len(h), half_life)
    sq = h ** 2
    num = (sq.fillna(0.0).values * w[:, None]).sum(axis=0); den = (sq.notna().values * w[:, None]).sum(axis=0)
    v = pd.Series(np.where(den > 0, num / np.maximum(den, 1e-12), np.nan), index=h.columns)
    n = h.notna().sum()
    med = float(np.nanmedian(v))
    shrink = (n / (n + MIN_HIST)).clip(0, 1)
    return (shrink * v.fillna(med) + (1 - shrink) * med)


def exposures(panel: pd.DataFrame, styles: list[str], sectors: list[str], m: pd.Timestamp) -> pd.DataFrame:
    g = panel[panel.month == m].set_index("ticker")
    X = pd.DataFrame(index=g.index)
    X["market"] = 1.0
    for s in styles:
        X[s] = g[s].fillna(0.0)
    for s in sectors:
        X[f"ind:{s}"] = (g.sector == s).astype(float)
    return X


def decompose(w: pd.Series, X: pd.DataFrame, F: pd.DataFrame, D: pd.Series, styles: list[str]) -> dict:
    """Annualised risk of weights w (active or absolute) under the model, by source."""
    names = [t for t in w.index if t in X.index]
    wv = w.reindex(names).fillna(0.0)
    Xn = X.loc[names, F.index].fillna(0.0)
    x = Xn.T @ wv
    var_f = float(x @ F.values @ x)
    dv = D.reindex(names).fillna(float(D.median()))
    var_s = float((wv ** 2 * dv).sum())
    tot = var_f + var_s
    groups = {"market": ["market"], "style": [s for s in styles if s in F.index], "industry": [c for c in F.index if c.startswith("ind:")]}
    contrib = {}
    for gname, cols in groups.items():
        if not cols: continue
        xg = x.reindex(cols).fillna(0.0)
        contrib[gname] = float(xg @ F.loc[cols, cols].values @ xg)              # own-group variance (cross terms left in 'interaction')
    inter = var_f - sum(contrib.values())
    top = sorted(((c, float(x[c]), float(x[c] ** 2 * F.loc[c, c])) for c in F.index), key=lambda z: -abs(z[2]))[:8]
    ann = lambda v: float(np.sqrt(max(v, 0.0) * 12))
    return dict(total=ann(tot), factor=ann(var_f), specific=ann(var_s), share_factor=var_f / tot if tot > 0 else np.nan,
                groups={k: v / tot if tot > 0 else np.nan for k, v in contrib.items()}, interaction=inter / tot if tot > 0 else np.nan,
                exposures={c: float(x[c]) for c in F.index}, top=[dict(factor=c, exposure=e, var_share=v / tot if tot > 0 else np.nan) for c, e, v in top],
                n_names=int((wv != 0).sum()))


def bias_test(model: dict, n_port: int = 100, names_per: int = 50, seed: int = 0, log=print) -> pd.DataFrame:
    """Random long-only equal-weight portfolios each month: realised next-month return ÷ predicted
    monthly volatility.  Bias statistic = std of that ratio (≈ 1 when calibrated)."""
    panel, fr, U, styles, secs = model["panel"], model["factor_returns"], model["residuals"], model["styles"], model["sectors"]
    rng = np.random.default_rng(seed)
    months = [m for m in sorted(panel.month.unique()) if (fr.index < m).sum() >= 24]
    rows = []
    for m in months:
        X = exposures(panel, styles, secs, m); F = factor_cov(fr, m); D = specific_var(U, m)
        g = panel[(panel.month == m) & panel.fwd.notna()].set_index("ticker")
        tick = [t for t in g.index if t in X.index]
        if len(tick) < names_per * 2:
            continue
        for k in range(n_port):
            pick = rng.choice(tick, names_per, replace=False)
            w = pd.Series(1.0 / names_per, index=pick)
            d = decompose(w, X, F, D, styles)
            sig_m = d["total"] / np.sqrt(12)
            rows.append(dict(month=m, kind="random", pred_vol_m=sig_m, realized=float(g.fwd.reindex(pick).mean())))
        # the whole universe equal-weighted, and the aggregate-book benchmark if weights are known
        w = pd.Series(1.0 / len(tick), index=tick); d = decompose(w, X, F, D, styles)
        rows.append(dict(month=m, kind="universe", pred_vol_m=d["total"] / np.sqrt(12), realized=float(g.fwd.reindex(tick).mean())))
    bt = pd.DataFrame(rows); bt["z"] = bt.realized / bt.pred_vol_m
    log(f"bias test: {len(months)} months × {n_port} random portfolios; bias statistic random {bt[bt.kind == 'random'].z.std():.2f}, universe {bt[bt.kind == 'universe'].z.std():.2f}")
    return bt


def build(out_dir: Path = OUT_DIR, log=print) -> dict:
    model = build_model(log)
    fr, U, panel, styles, secs = model["factor_returns"], model["residuals"], model["panel"], model["styles"], model["sectors"]
    out_dir.mkdir(parents=True, exist_ok=True)
    fr.to_csv(out_dir / "factor_returns.csv", index_label="month")
    model["r2"].to_csv(out_dir / "r2_monthly.csv", index_label="month", header=["r2"])
    asof = fr.index.max()
    F = factor_cov(fr, asof); D = specific_var(U, asof)
    F.to_csv(out_dir / "factor_covariance_latest.csv", index_label="factor")
    vols = pd.DataFrame({"factor": F.index, "vol_ann": np.sqrt(np.diag(F.values) * 12), "mean_ret_ann": fr.tail(WINDOW).mean().reindex(F.index).values * 12,
                         "t": (fr.tail(WINDOW).mean() / fr.tail(WINDOW).std() * np.sqrt(min(WINDOW, len(fr)))).reindex(F.index).values})
    vols.to_csv(out_dir / "factor_vols_latest.csv", index=False)
    pd.DataFrame({"ticker": D.index, "specific_vol_ann": np.sqrt(D.values * 12)}).sort_values("specific_vol_ann", ascending=False).to_csv(out_dir / "specific_risk_latest.csv", index=False)
    # decompositions at the latest month: universe equal-weight, aggregate book (from construction inputs), LP active weights
    m_last = panel.month.max()
    X = exposures(panel, styles, secs, m_last)
    decs = {}
    uni = list(X.index); decs["Equal-weight universe"] = decompose(pd.Series(1.0 / len(uni), index=uni), X, F, D, styles)
    inp = ROOT / "data" / "research" / "construction" / "inputs_latest"
    if (inp / "universe.csv").exists():
        u = pd.read_csv(inp / "universe.csv").set_index("ticker")
        decs["Aggregate book (benchmark)"] = decompose(u.bench, X, F, D, styles)
        if (inp / "weights_python.csv").exists():
            wp = pd.read_csv(inp / "weights_python.csv").set_index("ticker").weight
            decs["LP portfolio"] = decompose(wp, X, F, D, styles)
            act = wp.reindex(u.index.union(wp.index)).fillna(0) - u.bench.reindex(u.index.union(wp.index)).fillna(0)
            decs["LP portfolio — active vs benchmark"] = decompose(act, X, F, D, styles)
    (out_dir / "decompositions_latest.json").write_text(json.dumps(decs, indent=1, default=float))
    bt = bias_test(model, log=log)
    bt.to_csv(out_dir / "bias_test.csv", index=False)
    bias = {k: float(bt[bt.kind == k].z.std()) for k in bt.kind.unique()}
    # rolling bias by year for the random portfolios
    by_year = bt[bt.kind == "random"].groupby(bt.month.dt.year).z.std()
    by_year.to_csv(out_dir / "bias_by_year.csv", index_label="year", header=["bias_stat"])
    man = dict(built=str(date.today()), months=int(len(fr)), first=str(fr.index.min().date()), last=str(fr.index.max().date()), factors=int(len(F)),
               styles=styles, industries=len(secs), r2_avg=float(model["r2"].mean()), half_life=HALF_LIFE, window=WINDOW, bias=bias,
               universe_avg=model["info"]["universe_avg"], latest_month=str(m_last.date()))
    (out_dir / "manifest.json").write_text(json.dumps(man, indent=1, default=str))
    log(f"wrote {out_dir}")
    return man
