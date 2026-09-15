"""Fund-of-funds construction: combine the external managers that pass.

The manager-research job does not end at the memo.  Given N candidate managers, each
with an estimated alpha and a track record, the allocator has to decide how much of
each to hold — and the naive answer (weight by past alpha) is wrong for a well-known
reason: the alphas are noisy, and the noisiest ones look best.

    1  Shrink.  Each manager's FF3 alpha α̂ with standard error s is pulled toward the
       cross-sectional prior: α* = α̂ · τ² / (τ² + s²), with τ² = max(0, Var(α̂) − mean(s²))
       estimated across the universe (empirical Bayes).  A 6% alpha with a 5% standard
       error becomes a 1% alpha; a 2% alpha with a 0.8% standard error keeps most of it.
    2  Correlate.  Managers' *residual* returns (what is left after the factors) are what
       diversification acts on.  Pairwise correlations are estimated on overlapping months
       and shrunk halfway to zero.
    3  Allocate.  Maximise the blend's expected information ratio α*ᵀw / √(wᵀΣw) over
       long-only weights with a cap per manager; compare with equal weight of the top
       names and with holding everyone.
    4  Test it honestly.  Select on the first half of the sample, evaluate on the second:
       does picking managers by past alpha produce future alpha?  (The literature says
       barely, and the page reports whatever it finds.)

Only 13F clones and listed funds are available here, so this is the machinery on public
proxies; with real manager series (net of fees, audited) the same code applies.
Outputs: data/research/fund-of-funds/.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "research" / "fund-of-funds"
MIN_MONTHS = 60
MAX_WEIGHT = 0.25
CORR_SHRINK = 0.5
NOT_MEANINGFUL = {"multi", "macro", "mm"}
FACS = ["MKT_RF", "SMB", "HML"]


def load_managers(log=print) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Per-manager FF3 stats, the panel of monthly FF3 residuals (month × manager), and the aligned data."""
    rows, resid, AL = [], {}, {}
    cands = [(d.name, "fund", ROOT / "data" / "funds" / d.name) for d in sorted((ROOT / "output" / "funds").glob("*")) if d.is_dir()]
    cands += [(d.name, "listed", ROOT / "data" / "tickers" / d.name) for d in sorted((ROOT / "output" / "t").glob("*")) if d.is_dir()] if (ROOT / "output" / "t").exists() else []
    for key, kind, ddir in cands:
        out = ROOT / "output" / ("funds" if kind == "fund" else "t") / key
        al = out / "phase4" / "aligned_data.csv"
        if not al.exists():
            continue
        meta = json.loads((ddir / "meta.json").read_text()) if (ddir / "meta.json").exists() else {}
        style = meta.get("style", "listed" if kind == "listed" else "")
        if style in NOT_MEANINGFUL:
            continue
        d = pd.read_csv(al).dropna(subset=["y"] + FACS)
        if len(d) < MIN_MONTHS or not str(d.cell.iloc[0]).count("-"):      # monthly grids only
            continue
        X = sm.add_constant(d[FACS].values)
        f = sm.OLS(d.y.values, X).fit(cov_type="HAC", cov_kwds={"maxlags": int(np.floor(0.75 * len(d) ** (1 / 3)))})
        u = pd.Series(f.resid, index=pd.PeriodIndex(d.cell, freq="M").to_timestamp("M"))
        resid[key] = u; AL[key] = d.assign(month=u.index)
        rows.append(dict(key=key, name=meta.get("name", key), manager=meta.get("manager", ""), style=meta.get("style_name", "Listed fund" if kind == "listed" else style),
                         kind=kind, months=len(d), first=str(d.cell.iloc[0]), last=str(d.cell.iloc[-1]),
                         alpha=f.params[0] * 12, se=f.bse[0] * 12, t=f.tvalues[0], resid_vol=float(u.std() * np.sqrt(12)),
                         beta=f.params[1], smb=f.params[2], hml=f.params[3]))
    M = pd.DataFrame(rows).set_index("key")
    U = pd.DataFrame(resid)
    log(f"fund of funds: {len(M)} candidates with ≥ {MIN_MONTHS} months; {U.index.min().date()} → {U.index.max().date()}")
    return M, U, AL


def shrink_alphas(M: pd.DataFrame, tau: float | None = None) -> pd.DataFrame:
    """Empirical Bayes by default: prior mean = precision-weighted cross-sectional mean, prior variance
    τ² = Var(α̂) − mean(s²) (floored at zero).  Pass tau to impose a subjective prior instead
    (Baks–Metrick–Wachter: how much cross-sectional alpha dispersion you are willing to believe)."""
    prec = 1.0 / M.se ** 2
    mu = float((M.alpha * prec).sum() / prec.sum())
    tau2 = float(tau ** 2) if tau is not None else max(0.0, float(M.alpha.var(ddof=1) - (M.se ** 2).mean()))
    k = tau2 / (tau2 + M.se ** 2) if tau2 > 0 else 0.0 * M.se
    M = M.copy(); M["shrink_factor"] = k; M["alpha_shrunk"] = mu + k * (M.alpha - mu); M["tau"] = np.sqrt(tau2); M["prior_mean"] = mu
    M["ir_raw"] = M.alpha / M.resid_vol; M["ir_shrunk"] = M.alpha_shrunk / M.resid_vol
    return M


def residual_cov(U: pd.DataFrame, keys: list[str], shrink: float = CORR_SHRINK) -> pd.DataFrame:
    u = U[keys]
    C = u.corr(min_periods=24).fillna(0.0)
    np.fill_diagonal(C.values, 1.0)
    C = (1 - shrink) * C + shrink * np.eye(len(keys))
    sd = u.std() * np.sqrt(12)
    return C * np.outer(sd, sd)


def allocate(alpha: pd.Series, S: pd.DataFrame, max_weight: float = MAX_WEIGHT) -> pd.Series:
    """Long-only weights maximising expected IR = αᵀw / √(wᵀSw), Σw = 1, w ≤ cap."""
    keys = list(alpha.index); a = alpha.values; Sv = S.loc[keys, keys].values; n = len(keys)
    if n == 0:
        return pd.Series(dtype=float)
    def neg_ir(w):
        v = float(w @ Sv @ w); return -float(a @ w) / np.sqrt(max(v, 1e-12))
    w0 = np.full(n, 1.0 / n)
    res = minimize(neg_ir, w0, method="SLSQP", bounds=[(0.0, max_weight)] * n,
                   constraints=[dict(type="eq", fun=lambda w: w.sum() - 1.0)], options=dict(maxiter=500, ftol=1e-12))
    w = pd.Series(np.clip(res.x, 0, None), index=keys); w[w < 1e-4] = 0.0
    return w / w.sum()


def blend_stats(w: pd.Series, alpha: pd.Series, S: pd.DataFrame) -> dict:
    keys = list(w.index); wv = w.values
    a = float(alpha.reindex(keys).values @ wv); v = float(wv @ S.loc[keys, keys].values @ wv)
    return dict(alpha=a, vol=float(np.sqrt(v)), ir=a / np.sqrt(v) if v > 0 else np.nan, names=int((w > 0).sum()))


def diversification_curve(M: pd.DataFrame, S: pd.DataFrame, max_n: int = 25) -> pd.DataFrame:
    order = list(M.sort_values("ir_shrunk", ascending=False).index)
    rows = []
    for n in range(1, min(max_n, len(order)) + 1):
        keys = order[:n]; w = pd.Series(1.0 / n, index=keys)
        st = blend_stats(w, M.alpha_shrunk, S)
        rows.append(dict(n=n, added=keys[-1], **st))
    return pd.DataFrame(rows)


def out_of_sample(AL: dict, top: int = 10, log=print) -> dict:
    """Rank managers by FF3 alpha fitted on the first part of the sample; measure the alpha of the second
    part with the FIRST part's betas (a true out-of-sample residual — full-sample residuals would sum
    to zero per manager and fake a reversal).  Split month chosen so the most managers qualify."""
    from scipy import stats
    months = sorted({m for d in AL.values() for m in d.month})
    best, h = -1, None
    for cand in months[36:-24]:
        n = sum(1 for d in AL.values() if (d.month < cand).sum() >= 36 and (d.month >= cand).sum() >= 24)
        if n > best:
            best, h = n, cand
    if h is None:
        return dict(available=False, reason="no usable split")
    rows = {}
    for k, d in AL.items():
        a, b = d[d.month < h], d[d.month >= h]
        if len(a) < 36 or len(b) < 24:
            continue
        f = sm.OLS(a.y.values, sm.add_constant(a[FACS].values)).fit()
        u2 = b.y.values - sm.add_constant(b[FACS].values) @ np.r_[0.0, f.params[1:]]        # second-half returns net of first-half betas
        rows[k] = dict(alpha1=f.params[0] * 12, t1=f.tvalues[0], u2=pd.Series(u2, index=b.month.values))
    if len(rows) < 2 * top:
        return dict(available=False, reason=f"only {len(rows)} managers have enough history on both halves")
    t1 = pd.Series({k: v["t1"] for k, v in rows.items()}).sort_values(ascending=False)
    U2 = pd.DataFrame({k: v["u2"] for k, v in rows.items()})
    a1 = pd.Series({k: v["alpha1"] for k, v in rows.items()}); a2 = U2.mean() * 12
    groups = {"top": list(t1.index[:top]), "bottom": list(t1.index[-top:]), "all": list(rows)}
    out = dict(available=True, split=str(pd.Timestamp(h).date()), candidates=len(rows), n_top=top)
    for g, keys in groups.items():
        r = U2[keys].mean(axis=1).dropna()
        out[g] = dict(alpha=float(r.mean() * 12), vol=float(r.std() * np.sqrt(12)), ir=float(r.mean() / r.std() * np.sqrt(12)) if r.std() > 0 else np.nan,
                      t=float(r.mean() / r.std() * np.sqrt(len(r))) if r.std() > 0 else np.nan, months=int(len(r)), first_half_alpha=float(a1[keys].mean()))
    sp = (U2[groups["top"]].mean(axis=1) - U2[groups["bottom"]].mean(axis=1)).dropna()
    out["top_minus_bottom"] = dict(alpha=float(sp.mean() * 12), t=float(sp.mean() / sp.std() * np.sqrt(len(sp))) if sp.std() > 0 else np.nan, months=int(len(sp)))
    out["rank_corr"] = float(stats.spearmanr(a1, a2.reindex(a1.index)).statistic)
    out["share_positive_second"] = float((a2 > 0).mean())
    log(f"  out of sample (split {out['split']}, {len(rows)} managers): top {top} by first-half t → second-half alpha {out['top']['alpha']:+.1%} (t {out['top']['t']:+.1f}); "
        f"bottom {out['bottom']['alpha']:+.1%}; spread t {out['top_minus_bottom']['t']:+.1f}; rank corr {out['rank_corr']:+.2f}")
    return out


PRIORS = [0.01, 0.02, 0.04]      # subjective τ: cross-sectional alpha dispersion an allocator is willing to assume


def build(out_dir: Path = OUT_DIR, log=print) -> dict:
    M0, U, AL = load_managers(log)
    Meb = shrink_alphas(M0)                                  # empirical Bayes
    S = residual_cov(U, list(M0.index))
    tau_eb = float(Meb.tau.iloc[0])
    # allocations under the empirical-Bayes prior and under each subjective prior
    alloc = {}
    for label, tau in [("empirical Bayes", None)] + [(f"τ = {t:.0%}", t) for t in PRIORS]:
        M = shrink_alphas(M0, tau)
        sel = M[M.alpha_shrunk > 0]
        w = allocate(sel.alpha_shrunk, S) if len(sel) else pd.Series(dtype=float); w = w[w > 0]
        alloc[label] = dict(tau=float(M.tau.iloc[0]), positive=int(len(sel)), weights=w.to_dict(), **blend_stats(w, M.alpha_shrunk, S) if len(w) else dict(alpha=0.0, vol=0.0, ir=np.nan, names=0))
    # the displayed case: the moderate believer (τ = 2%), with the empirical-Bayes answer stated alongside
    M = shrink_alphas(M0, 0.02)
    w_mv = pd.Series(alloc["τ = 2%"]["weights"])
    top10 = list(M.sort_values("ir_shrunk", ascending=False).index[:10])
    w_eq = pd.Series(0.1, index=top10)
    w_all = pd.Series(1.0 / len(M), index=M.index)
    blends = {"Optimised (max IR, ≤ 25% each)": blend_stats(w_mv, M.alpha_shrunk, S) if len(w_mv) else dict(alpha=0, vol=0, ir=np.nan, names=0),
              "Equal weight, top 10 by shrunk IR": blend_stats(w_eq, M.alpha_shrunk, S), "Equal weight, everyone": blend_stats(w_all, M.alpha_shrunk, S)}
    curve = diversification_curve(M, S)
    oos = out_of_sample(AL, log=log)
    out_dir.mkdir(parents=True, exist_ok=True)
    M["alpha_eb"] = Meb.alpha_shrunk; M["weight_optimised"] = w_mv.reindex(M.index).fillna(0.0); M["weight_top10"] = w_eq.reindex(M.index).fillna(0.0)
    M.sort_values("alpha_shrunk", ascending=False).to_csv(out_dir / "managers.csv", index_label="key")
    (out_dir / "allocations_by_prior.json").write_text(json.dumps(alloc, indent=1, default=float))
    C = U[list(M.index)].corr(min_periods=24); C.to_csv(out_dir / "residual_correlation.csv", index_label="key")
    curve.to_csv(out_dir / "diversification_curve.csv", index=False)
    (out_dir / "blends.json").write_text(json.dumps(blends, indent=1, default=float))
    (out_dir / "out_of_sample.json").write_text(json.dumps(oos, indent=1, default=float))
    tri = C.values[np.triu_indices(len(C), 1)]
    n_sig = int((M.t >= 2).sum()); expect = len(M) * 0.023
    man = dict(built=str(date.today()), managers=int(len(M)), selected=int((w_mv > 0).sum()), positive_after_shrink=int((M.alpha_shrunk > 0).sum()),
               tau=float(M.tau.iloc[0]), tau_eb=tau_eb, prior_mean=float(M.prior_mean.iloc[0]), avg_shrink=float(M.shrink_factor.mean()), priors=PRIORS,
               alloc_summary={k: dict(tau=v["tau"], names=v["names"], ir=v["ir"], alpha=v["alpha"]) for k, v in alloc.items()}, avg_corr=float(np.nanmean(tri)), ir_blend=blends["Optimised (max IR, ≤ 25% each)"]["ir"],
               ir_top10=blends["Equal weight, top 10 by shrunk IR"]["ir"], ir_all=blends["Equal weight, everyone"]["ir"], n_t2=n_sig, expected_t2_by_chance=expect,
               first=str(U.index.min().date()), last=str(U.index.max().date()), min_months=MIN_MONTHS, max_weight=MAX_WEIGHT, corr_shrink=CORR_SHRINK,
               oos=oos)
    (out_dir / "manifest.json").write_text(json.dumps(man, indent=1, default=str))
    log(f"wrote {out_dir}")
    return man
