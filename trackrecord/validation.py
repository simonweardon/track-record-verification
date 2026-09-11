"""Phase 4 — statistical validation.  Does the excess return mean anything?

All tests run on the composite gross series over its cells (calendar years
now; months once monthly data exists) against Ken French factors on the same
cells.  Developed-market factors are primary (global mandate); the US set is
run as a robustness check.

1. Factor regressions   CAPM · FF3 · Carhart 4 · FF5
   R_p − RF = α + Σ b_k f_k + ε.   α with SE, t, p, 95% CI, loadings, R², n.
   Newey-West HAC standard errors for monthly cells (lag ⌊0.75·T^(1/3)⌋);
   plain OLS for annual cells (no meaningful serial correlation to correct).
2. Risk-adjusted metrics   Sharpe, Sortino (MAR = RF), information ratio,
   tracking error, max drawdown (understated on annual cells), Calmar,
   up/down capture, beta, correlation.  Benchmark metrics alongside.
3. Bootstrap
   a) Null bootstrap of t(α) (Kosowski et al. 2006 / Fama-French 2010 style):
      subtract the fitted α, resample cells with replacement, refit; p = share
      of simulated t(α) ≥ observed.  Plus a pairs-bootstrap percentile CI for α.
   b) Random-manager cohort: N zero-skill managers with the same beta and the
      same residual volatility over the same market path; report the
      percentile of the actual record's headline excess return and of its t(α).
      Variant: market path resampled too.
4. Rolling windows   Configurable; on annual cells 5-cell rolling excess return
   (descriptive) and 10-cell rolling CAPM α with CI (weak inference).  On
   monthly cells 36 and 60.
5. Sub-periods   α before/after a split (default 2010) via an interaction
   regression; test that the two αs are equal.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

FACTOR_SETS = {
    "DEV": dict(MKT_RF="DEV_MKT_RF", SMB="DEV_SMB", HML="DEV_HML", MOM="DEV_MOM", RMW="DEV_RMW", CMA="DEV_CMA", RF="DEV_RF"),
    "US":  dict(MKT_RF="US_MKT_RF",  SMB="US_SMB",  HML="US_HML",  MOM="US_MOM",  RMW="US_RMW",  CMA="US_CMA",  RF="US_RF"),
}
MODELS = {
    "CAPM":      ["MKT_RF"],
    "FF3":       ["MKT_RF", "SMB", "HML"],
    "Carhart4":  ["MKT_RF", "SMB", "HML", "MOM"],
    "FF5":       ["MKT_RF", "SMB", "HML", "RMW", "CMA"],
}


@dataclass
class ValidationConfig:
    periods_per_year: int = 1          # 1 annual, 12 monthly
    factor_set: str = "DEV"
    robustness_set: str = "US"
    n_boot: int = 5000
    n_cohort: int = 10000
    split_year: int = 2010
    seed: int = 0
    rolling_desc: int | None = None    # default 5 cells (annual) / 36 (monthly)
    rolling_inf: int | None = None     # default 10 cells (annual) / 60 (monthly)


@dataclass
class Phase4Result:
    data: pd.DataFrame                   # aligned excess returns + factors, per cell
    regressions: pd.DataFrame
    metrics: pd.DataFrame
    bootstrap: pd.DataFrame
    cohort: pd.DataFrame
    cohort_samples: pd.DataFrame          # simulated headline excess per manager, by variant
    rolling: pd.DataFrame
    subperiods: pd.DataFrame
    risk_by_year: pd.DataFrame
    risk_return: pd.DataFrame
    config: ValidationConfig
    notes: list[str] = field(default_factory=list)


# ---- data ----------------------------------------------------------------------

def align(comp: pd.DataFrame, factors: pd.DataFrame, fs: dict) -> pd.DataFrame:
    """comp indexed by cell (year or Period); factors indexed the same way."""
    df = pd.DataFrame(index=comp.index)
    df["r_p"] = comp.gross
    df["r_b"] = comp.benchmark
    fac = factors
    if isinstance(comp.index, pd.PeriodIndex) and not isinstance(factors.index, pd.PeriodIndex):
        fac = factors.copy(); fac.index = pd.DatetimeIndex(fac.index).to_period("M")
    for k, col in fs.items():
        df[k] = fac[col].reindex(comp.index) if col in fac else np.nan
    df["y"] = df.r_p - df.RF
    df["y_b"] = df.r_b - df.RF
    return df


def _fit(y: pd.Series, X: pd.DataFrame, ppy: int):
    Xc = sm.add_constant(X.values)
    if ppy > 1:
        lag = int(np.floor(0.75 * len(y) ** (1 / 3)))
        return sm.OLS(y.values, Xc).fit(cov_type="HAC", cov_kwds={"maxlags": lag}), lag
    return sm.OLS(y.values, Xc).fit(), 0


def factor_regressions(df: pd.DataFrame, cfg: ValidationConfig, set_name: str) -> pd.DataFrame:
    rows = []
    for model, facs in MODELS.items():
        sub = df[["y"] + facs].dropna()
        if len(sub) < len(facs) + 3:
            rows.append(dict(factor_set=set_name, model=model, n=len(sub), note="insufficient observations"))
            continue
        fit, lag = _fit(sub.y, sub[facs], cfg.periods_per_year)
        ppy = cfg.periods_per_year
        a, se = fit.params[0], fit.bse[0]
        ci = fit.conf_int(alpha=0.05)[0]
        row = dict(factor_set=set_name, model=model, n=len(sub), hac_lag=lag,
                   alpha_per_period=a, alpha_annual=a * ppy, se_annual=se * ppy,
                   t=fit.tvalues[0], p=fit.pvalues[0],
                   ci_low_annual=ci[0] * ppy, ci_high_annual=ci[1] * ppy,
                   r2=fit.rsquared, r2_adj=fit.rsquared_adj, resid_sd_annual=np.sqrt(fit.mse_resid) * np.sqrt(ppy))
        for i, f in enumerate(facs):
            row[f"b_{f}"] = fit.params[i + 1]; row[f"t_{f}"] = fit.tvalues[i + 1]
        rows.append(row)
    return pd.DataFrame(rows)


# ---- metrics ---------------------------------------------------------------------

def max_drawdown(r: pd.Series) -> float:
    idx = (1 + r.fillna(0)).cumprod()
    return float((idx / idx.cummax() - 1).min())


def risk_metrics(df: pd.DataFrame, ppy: int) -> pd.DataFrame:
    d = df.dropna(subset=["r_p", "r_b", "RF"])
    y, yb, act = d.y, d.y_b, d.r_p - d.r_b
    n = len(d); years = n / ppy
    def ann(r): return float((1 + r).prod() ** (1 / years) - 1)
    def dd(r):
        neg = np.minimum(r, 0)
        return float(np.sqrt((neg ** 2).mean()))
    up, dn = d.r_b > 0, d.r_b <= 0
    def cap(mask):
        if mask.sum() == 0: return np.nan
        gp = (1 + d.r_p[mask]).prod() ** (1 / mask.sum()) - 1
        gb = (1 + d.r_b[mask]).prod() ** (1 / mask.sum()) - 1
        return float(gp / gb) if gb != 0 else np.nan
    beta = float(np.cov(y, yb, ddof=1)[0, 1] / yb.var(ddof=1)) if yb.var(ddof=1) > 0 else np.nan
    rows = [
        dict(metric="periods", fmt="int", portfolio=n, benchmark=n),
        dict(metric="annualized return", fmt="pct", portfolio=ann(d.r_p), benchmark=ann(d.r_b)),
        dict(metric="annualized volatility", fmt="pct", portfolio=float(d.r_p.std(ddof=1) * np.sqrt(ppy)), benchmark=float(d.r_b.std(ddof=1) * np.sqrt(ppy))),
        dict(metric="Sharpe (excess over RF)", fmt="num", portfolio=float(y.mean() / y.std(ddof=1) * np.sqrt(ppy)), benchmark=float(yb.mean() / yb.std(ddof=1) * np.sqrt(ppy))),
        dict(metric="Sortino (MAR = RF)", fmt="num", portfolio=float(y.mean() / dd(y) * np.sqrt(ppy)) if dd(y) > 0 else np.nan,
             benchmark=float(yb.mean() / dd(yb) * np.sqrt(ppy)) if dd(yb) > 0 else np.nan),
        dict(metric="max drawdown (cell-level; understated on annual cells)", fmt="pct", portfolio=max_drawdown(d.r_p), benchmark=max_drawdown(d.r_b)),
        dict(metric="Calmar (ann. return / |max DD|)", fmt="num", portfolio=ann(d.r_p) / abs(max_drawdown(d.r_p)) if max_drawdown(d.r_p) < 0 else np.nan,
             benchmark=ann(d.r_b) / abs(max_drawdown(d.r_b)) if max_drawdown(d.r_b) < 0 else np.nan),
        dict(metric="tracking error (ann.)", fmt="pct", portfolio=float(act.std(ddof=1) * np.sqrt(ppy)), benchmark=np.nan),
        dict(metric="information ratio", fmt="num", portfolio=float(act.mean() / act.std(ddof=1) * np.sqrt(ppy)) if act.std(ddof=1) > 0 else np.nan, benchmark=np.nan),
        dict(metric="beta vs benchmark", fmt="num", portfolio=beta, benchmark=1.0),
        dict(metric="correlation vs benchmark", fmt="num", portfolio=float(np.corrcoef(d.r_p, d.r_b)[0, 1]), benchmark=1.0),
        dict(metric="up capture", fmt="num", portfolio=cap(up), benchmark=1.0),
        dict(metric="down capture", fmt="num", portfolio=cap(dn), benchmark=1.0),
        dict(metric="periods benchmark was up / down", fmt="str", portfolio=f"{int(up.sum())} / {int(dn.sum())}", benchmark=""),
        dict(metric="variance (ann.)", fmt="pct", portfolio=float(d.r_p.var(ddof=1) * ppy), benchmark=float(d.r_b.var(ddof=1) * ppy)),
        dict(metric=f"VaR 95%, one {'month' if ppy == 12 else 'year'}, historical", fmt="pct", portfolio=var_hist(d.r_p), benchmark=var_hist(d.r_b)),
        dict(metric=f"VaR 95%, one {'month' if ppy == 12 else 'year'}, parametric", fmt="pct", portfolio=var_param(d.r_p), benchmark=var_param(d.r_b)),
        dict(metric=f"expected shortfall 95%, one {'month' if ppy == 12 else 'year'}", fmt="pct", portfolio=es_hist(d.r_p), benchmark=es_hist(d.r_b)),
        dict(metric=f"worst {'month' if ppy == 12 else 'year'}", fmt="pct", portfolio=float(d.r_p.min()), benchmark=float(d.r_b.min())),
    ]
    return pd.DataFrame(rows)


# ---- tail risk helpers (loss conventions: VaR is a return level, negative = loss) --------

def var_hist(r: pd.Series, q: float = 0.05) -> float:
    r = r.dropna()
    return float(np.percentile(r, q * 100)) if len(r) >= 3 else np.nan


def var_param(r: pd.Series, q: float = 0.05) -> float:
    r = r.dropna()
    return float(r.mean() + stats.norm.ppf(q) * r.std(ddof=1)) if len(r) >= 3 else np.nan


def es_hist(r: pd.Series, q: float = 0.05) -> float:
    r = r.dropna()
    if len(r) < 3:
        return np.nan
    cut = np.percentile(r, q * 100)
    tail = r[r <= cut]
    return float(tail.mean()) if len(tail) else np.nan


def risk_by_year(df: pd.DataFrame, ppy: int, trailing: int | None = None) -> pd.DataFrame:
    """Per calendar year: compounded return, and — when the cells are months — the
    within-year volatility, variance, 95% VaR (historical and parametric), expected
    shortfall and worst month, for portfolio and benchmark.  With annual cells the
    within-year figures are not observable (one number per year); a trailing-window
    version over `trailing` cells (default 60 months / 10 years) is given for both grids."""
    d = df.dropna(subset=["r_p", "r_b"]).copy()
    d["year"] = [c if isinstance(c, (int, np.integer)) else c.year for c in d.index]
    trailing = trailing or (60 if ppy == 12 else 10)
    rows = []
    for y, g in d.groupby("year"):
        row = dict(year=int(y), n=len(g),
                   return_p=float((1 + g.r_p).prod() - 1), return_b=float((1 + g.r_b).prod() - 1))
        for side, col in [("p", "r_p"), ("b", "r_b")]:
            r = g[col]
            if ppy > 1 and len(r) >= 3:
                sd = float(r.std(ddof=1))
                row[f"vol_{side}"] = sd * np.sqrt(ppy)
                row[f"variance_{side}"] = sd ** 2 * ppy
                row[f"var95_hist_{side}"] = var_hist(r)
                row[f"var95_param_{side}"] = var_param(r)
                row[f"es95_{side}"] = es_hist(r)
                row[f"worst_{side}"] = float(r.min())
            else:
                for k in ["vol", "variance", "var95_hist", "var95_param", "es95", "worst"]:
                    row[f"{k}_{side}"] = np.nan
            # trailing window ending at this year's last cell
            upto = d.loc[:g.index[-1], col].tail(trailing)
            if len(upto) >= max(3, trailing // 2):
                row[f"trailing_vol_{side}"] = float(upto.std(ddof=1) * np.sqrt(ppy))
                row[f"trailing_var95_hist_{side}"] = var_hist(upto)
                row[f"trailing_es95_{side}"] = es_hist(upto)
            else:
                row[f"trailing_vol_{side}"] = row[f"trailing_var95_hist_{side}"] = row[f"trailing_es95_{side}"] = np.nan
        rows.append(row)
    out = pd.DataFrame(rows).set_index("year")
    out.attrs["trailing"] = trailing
    out.attrs["within_year_observable"] = ppy > 1
    return out


def risk_return_points(comp: pd.DataFrame, df: pd.DataFrame, ppy: int,
                       account_periods: pd.DataFrame | None = None) -> pd.DataFrame:
    """Annualized return vs annualized volatility for the composite, benchmark,
    model-net, the risk-free rate, and every account over its eligible cells."""
    def ann(r):
        r = r.dropna(); yrs = len(r) / ppy
        return float((1 + r).prod() ** (1 / yrs) - 1) if yrs > 0 else np.nan
    def vol(r):
        r = r.dropna()
        return float(r.std(ddof=1) * np.sqrt(ppy)) if len(r) >= 3 else np.nan
    rf = df.RF.dropna()
    rows = [dict(name="Composite (gross)", kind="composite", n=int(comp.gross.notna().sum()), ret=ann(comp.gross), vol=vol(comp.gross)),
            dict(name="Benchmark", kind="benchmark", n=int(comp.benchmark.notna().sum()), ret=ann(comp.benchmark), vol=vol(comp.benchmark)),
            dict(name="Model-net", kind="modelnet", n=int(comp.model_net.notna().sum()), ret=ann(comp.model_net), vol=vol(comp.model_net)),
            dict(name="Risk-free", kind="rf", n=len(rf), ret=ann(rf), vol=0.0)]
    if account_periods is not None and len(account_periods):
        el = account_periods[account_periods.eligible == True]
        for acct, g in el.groupby("account_id"):
            r = g.sort_values("cell").r
            if len(r) >= 3:
                rows.append(dict(name=str(acct), kind="account", n=len(r), ret=ann(r), vol=vol(r),
                                 owner=str(g.owner_type.iloc[0]) if "owner_type" in g else ""))
    out = pd.DataFrame(rows)
    rf_ann = out.loc[out.kind == "rf", "ret"].iloc[0]
    out["sharpe"] = (out.ret - rf_ann) / out.vol.replace(0, np.nan)
    return out


# ---- bootstrap -------------------------------------------------------------------

def null_bootstrap(df: pd.DataFrame, model: str, cfg: ValidationConfig) -> dict:
    facs = MODELS[model]
    sub = df[["y"] + facs].dropna()
    X = sm.add_constant(sub[facs].values); y = sub.y.values
    fit = sm.OLS(y, X).fit()
    a_hat, t_obs = fit.params[0], fit.tvalues[0]
    y0 = y - a_hat                     # impose the null: alpha = 0
    rng = np.random.default_rng(cfg.seed)
    n = len(y); T = np.empty(cfg.n_boot); A = np.empty(cfg.n_boot)
    XtX_cache = None
    for b in range(cfg.n_boot):
        idx = rng.integers(0, n, n)
        fb = sm.OLS(y0[idx], X[idx]).fit()
        T[b] = fb.tvalues[0]
        fa = sm.OLS(y[idx], X[idx]).fit()   # pairs bootstrap without the null, for a CI
        A[b] = fa.params[0]
    ppy = cfg.periods_per_year
    return dict(model=model, n=n, alpha_annual=a_hat * ppy, t_observed=t_obs,
                p_null_one_sided=float((T >= t_obs).mean()),
                p_null_two_sided=float((np.abs(T) >= abs(t_obs)).mean()),
                alpha_ci_low_annual=float(np.percentile(A, 2.5)) * ppy,
                alpha_ci_high_annual=float(np.percentile(A, 97.5)) * ppy,
                share_boot_alpha_below_zero=float((A <= 0).mean()), n_boot=cfg.n_boot)


def random_cohort(df: pd.DataFrame, cfg: ValidationConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Zero-skill managers with the record's own beta and residual volatility."""
    sub = df[["y", "MKT_RF", "RF", "r_b"]].dropna()
    x = sub.MKT_RF.values; y = sub.y.values; rf = sub.RF.values; rb = sub.r_b.values
    fit = sm.OLS(y, sm.add_constant(x)).fit()
    beta, sig = fit.params[1], np.sqrt(fit.mse_resid)
    resid = fit.resid - fit.resid.mean()
    n = len(y); ppy = cfg.periods_per_year; years = n / ppy
    rng = np.random.default_rng(cfg.seed + 1)

    def headline_excess(rp, rbb):
        return (1 + rp).prod() ** (1 / years) - (1 + rbb).prod() ** (1 / years)

    actual_excess = headline_excess(sub.r_p.values if "r_p" in sub else y + rf, rb)
    actual_t = fit.tvalues[0]
    out = []; samples = {}
    for variant in ["same market path", "market path resampled"]:
        ex = np.empty(cfg.n_cohort); ts = np.empty(cfg.n_cohort)
        for k in range(cfg.n_cohort):
            if variant == "same market path":
                xi, rfi, rbi = x, rf, rb
            else:
                idx = rng.integers(0, n, n); xi, rfi, rbi = x[idx], rf[idx], rb[idx]
            e = rng.choice(resid, n, replace=True)          # bootstrap residuals (no normality assumed)
            yi = beta * xi + e                               # alpha = 0
            rpi = yi + rfi
            ex[k] = headline_excess(rpi, rbi)
            f = sm.OLS(yi, sm.add_constant(xi)).fit()
            ts[k] = f.tvalues[0]
        samples[variant] = ex
        out.append(dict(variant=variant, n_managers=cfg.n_cohort, beta=beta, resid_sd_annual=sig * np.sqrt(ppy),
                        actual_headline_excess=actual_excess,
                        percentile_by_headline_excess=float((ex < actual_excess).mean() * 100),
                        cohort_excess_p5=float(np.percentile(ex, 5)), cohort_excess_p50=float(np.percentile(ex, 50)),
                        cohort_excess_p95=float(np.percentile(ex, 95)),
                        actual_t_alpha=actual_t, percentile_by_t_alpha=float((ts < actual_t).mean() * 100),
                        share_of_zero_skill_managers_beating_actual=float((ex >= actual_excess).mean())))
    return pd.DataFrame(out), pd.DataFrame(samples)


# ---- rolling and sub-periods ------------------------------------------------------

def rolling(df: pd.DataFrame, cfg: ValidationConfig) -> pd.DataFrame:
    ppy = cfg.periods_per_year
    w_desc = cfg.rolling_desc or (5 if ppy == 1 else 36)
    w_inf = cfg.rolling_inf or (10 if ppy == 1 else 60)
    d = df.dropna(subset=["r_p", "r_b", "RF", "MKT_RF"])
    rows = []
    idx = list(d.index)
    for i in range(len(d)):
        row = dict(cell=idx[i])
        if i + 1 >= w_desc:
            s = d.iloc[i + 1 - w_desc:i + 1]
            yrs = w_desc / ppy
            row[f"excess_{w_desc}"] = (1 + s.r_p).prod() ** (1 / yrs) - (1 + s.r_b).prod() ** (1 / yrs)
        if i + 1 >= w_inf:
            s = d.iloc[i + 1 - w_inf:i + 1]
            f = sm.OLS(s.y.values, sm.add_constant(s.MKT_RF.values)).fit()
            ci = f.conf_int(0.05)[0]
            row[f"alpha_{w_inf}"] = f.params[0] * ppy
            row[f"alpha_{w_inf}_ci_low"] = ci[0] * ppy
            row[f"alpha_{w_inf}_ci_high"] = ci[1] * ppy
            row[f"alpha_{w_inf}_t"] = f.tvalues[0]
            row[f"beta_{w_inf}"] = f.params[1]
        rows.append(row)
    return pd.DataFrame(rows).set_index("cell")


def subperiods(df: pd.DataFrame, cfg: ValidationConfig, model: str = "CAPM") -> pd.DataFrame:
    facs = MODELS[model]
    d = df[["y"] + facs].dropna()
    year = np.array([c if isinstance(c, (int, np.integer)) else c.year for c in d.index])
    D = (year >= cfg.split_year).astype(float)
    X = np.column_stack([np.ones(len(d)), D] + [d[f].values for f in facs] + [d[f].values * D for f in facs])
    f = sm.OLS(d.y.values, X).fit()
    ppy = cfg.periods_per_year
    a_pre, a_diff = f.params[0], f.params[1]
    se_pre = f.bse[0]
    # post alpha = a_pre + a_diff; its SE from the covariance matrix
    cov = f.cov_params()
    se_post = float(np.sqrt(cov[0, 0] + cov[1, 1] + 2 * cov[0, 1]))
    dfree = f.df_resid
    rows = [dict(model=model, segment=f"before {cfg.split_year}", n=int((D == 0).sum()), alpha_annual=a_pre * ppy,
                 se_annual=se_pre * ppy, t=a_pre / se_pre, p=2 * stats.t.sf(abs(a_pre / se_pre), dfree)),
            dict(model=model, segment=f"{cfg.split_year} onward", n=int((D == 1).sum()), alpha_annual=(a_pre + a_diff) * ppy,
                 se_annual=se_post * ppy, t=(a_pre + a_diff) / se_post, p=2 * stats.t.sf(abs((a_pre + a_diff) / se_post), dfree)),
            dict(model=model, segment="difference (post − pre)", n=len(d), alpha_annual=a_diff * ppy,
                 se_annual=f.bse[1] * ppy, t=f.tvalues[1], p=f.pvalues[1])]
    return pd.DataFrame(rows)


# ---- build + report -----------------------------------------------------------------

def build(comp: pd.DataFrame, factors: pd.DataFrame, cfg: ValidationConfig = ValidationConfig(),
          account_periods: pd.DataFrame | None = None) -> Phase4Result:
    notes = []
    fs = FACTOR_SETS[cfg.factor_set]
    df = align(comp, factors, fs)
    regs = [factor_regressions(df, cfg, cfg.factor_set)]
    if cfg.robustness_set and cfg.robustness_set != cfg.factor_set:
        df_us = align(comp, factors, FACTOR_SETS[cfg.robustness_set])
        regs.append(factor_regressions(df_us, cfg, cfg.robustness_set))
    regressions = pd.concat(regs, ignore_index=True)
    metrics = risk_metrics(df, cfg.periods_per_year)
    boot = pd.DataFrame([null_bootstrap(df, m, cfg) for m in ["CAPM", "FF3"]])
    cohort, cohort_samples = random_cohort(df, cfg)
    roll = rolling(df, cfg)
    sp = pd.concat([subperiods(df, cfg, "CAPM"), subperiods(df, cfg, "FF3")], ignore_index=True)
    rby = risk_by_year(df, cfg.periods_per_year)
    rr = risk_return_points(comp, df, cfg.periods_per_year, account_periods)
    n = int(df.y.notna().sum())
    capm = regressions[(regressions.factor_set == cfg.factor_set) & (regressions.model == "CAPM")]
    if len(capm) and not pd.isna(capm.iloc[0].get("resid_sd_annual", np.nan)):
        sd = float(capm.iloc[0].resid_sd_annual)
        w_inf = cfg.rolling_inf or (10 if cfg.periods_per_year == 1 else 60)
        se_w = sd / np.sqrt(w_inf / cfg.periods_per_year)
        notes.append(f"Noise floor: residual volatility is {sd:.1%}/yr, so a {w_inf}-cell rolling alpha has a "
                     f"standard error of about ±{se_w:.1%}/yr and will swing by ±{2 * se_w:.1%} even if true "
                     "alpha never changes. Read the rolling series against that yardstick, not against zero.")
    notes.append("Many tests are reported (4 models × 2 factor sets, sub-periods, rolling windows). At the 5% "
                 "level roughly one in twenty will look significant by chance; a single starred cell is not "
                 "evidence — consistency across models and periods is.")
    if cfg.periods_per_year == 1:
        notes.append(f"{n} annual observations. Alpha precision is driven by span, not frequency, so the "
                     "headline t-statistic is about as good as monthly data would give; what annual data "
                     "cannot support is short rolling windows, drawdown, and multi-factor loadings.")
        notes.append("3- and 5-year rolling alpha requires monthly data; shown here are a 5-year rolling "
                     "excess return (descriptive) and a 10-year rolling CAPM alpha (weak inference).")
    return Phase4Result(df, regressions, metrics, boot, cohort, cohort_samples, roll, sp, rby, rr, cfg, notes)


def _pct(x, d=2):
    return "n/a" if x is None or pd.isna(x) else f"{x * 100:+.{d}f}%"


def _stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def write_phase4(res: Phase4Result, out_dir, title="Phase 4 — statistical validation"):
    from pathlib import Path
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    res.data.to_csv(out / "aligned_data.csv"); res.regressions.to_csv(out / "regressions.csv", index=False)
    res.metrics.to_csv(out / "metrics.csv", index=False); res.bootstrap.to_csv(out / "bootstrap.csv", index=False)
    res.cohort.to_csv(out / "cohort.csv", index=False); res.rolling.to_csv(out / "rolling.csv")
    res.cohort_samples.to_csv(out / "cohort_samples.csv", index=False)
    res.risk_by_year.to_csv(out / "risk_by_year.csv"); res.risk_return.to_csv(out / "risk_return.csv", index=False)
    res.subperiods.to_csv(out / "subperiods.csv", index=False)
    cfg = res.config; R = res.regressions; ppy = cfg.periods_per_year
    unit = "year" if ppy == 1 else "month"

    L = [f"# {title}", ""]
    prim = R[(R.factor_set == cfg.factor_set) & (R.model == "CAPM")].iloc[0]
    L += ["## The one-paragraph answer", ""]
    verdict = ("statistically distinguishable from zero at the 5% level" if prim.p < 0.05 else
               "suggestive but not statistically distinguishable from zero at the 5% level" if prim.p < 0.15 else
               "not distinguishable from zero")
    L += [f"CAPM alpha vs {cfg.factor_set} market over {int(prim.n)} {unit}s: **{_pct(prim.alpha_annual)}/yr** "
          f"(95% CI {_pct(prim.ci_low_annual)} to {_pct(prim.ci_high_annual)}, t = {prim.t:.2f}, p = {prim.p:.3f}) — "
          f"{verdict}. Beta {prim.b_MKT_RF:.2f}, R² {prim.r2:.2f}.", ""]
    b = res.bootstrap[res.bootstrap.model == "CAPM"].iloc[0]
    c = res.cohort.iloc[0]
    L += [f"Null bootstrap: if true alpha were zero, a t-statistic this large would occur "
          f"**{b.p_null_one_sided:.1%}** of the time. Among {int(c.n_managers):,} simulated zero-skill managers "
          f"with the same beta and residual volatility facing the same markets, this record's headline excess "
          f"return ({_pct(c.actual_headline_excess)}/yr) sits at the **{c.percentile_by_headline_excess:.0f}th percentile**; "
          f"{c.share_of_zero_skill_managers_beating_actual:.1%} of them did at least as well by luck.", ""]

    L += ["## Factor regressions", "",
          f"Dependent variable: composite gross return minus RF, per {unit}. Alpha annualized"
          + (" (×12)" if ppy == 12 else "") + ". Standard errors: "
          + ("Newey-West HAC." if ppy > 1 else "OLS (annual cells).") + " Stars: * p<0.10 ** p<0.05 *** p<0.01.", "",
          "| factors | model | n | alpha/yr | 95% CI | t | p | β mkt | β SMB | β HML | β MOM | β RMW | β CMA | R² |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in R.iterrows():
        if "alpha_annual" not in r or pd.isna(r.get("alpha_annual")):
            L.append(f"| {r.factor_set} | {r.model} | {int(r.n)} | {r.get('note', '')} | | | | | | | | | | |"); continue
        def bb(f): return f"{r[f'b_{f}']:.2f}" if f"b_{f}" in r and not pd.isna(r[f"b_{f}"]) else "—"
        L.append(f"| {r.factor_set} | {r.model} | {int(r.n)} | **{_pct(r.alpha_annual)}**{_stars(r.p)} | "
                 f"{_pct(r.ci_low_annual, 1)} … {_pct(r.ci_high_annual, 1)} | {r.t:.2f} | {r.p:.3f} | "
                 f"{bb('MKT_RF')} | {bb('SMB')} | {bb('HML')} | {bb('MOM')} | {bb('RMW')} | {bb('CMA')} | {r.r2:.2f} |")
    L += ["", "Reading the loadings: β mkt ≈ 1 with small other loadings means the return is mostly the "
          "market; a large SMB/HML/MOM loading means part of what looks like alpha is a known style "
          "premium that an index fund could have delivered.", ""]

    L += ["## Risk-adjusted metrics", "", "| metric | portfolio | benchmark |", "|---|---|---|"]
    for _, m in res.metrics.iterrows():
        def fmt(v):
            if m.fmt == "str": return str(v)
            if v is None or (not isinstance(v, str) and pd.isna(v)): return "n/a"
            return {"pct": _pct(v), "num": f"{v:.2f}", "int": f"{int(v)}"}[m.fmt]
        L.append(f"| {m.metric} | {fmt(m.portfolio)} | {fmt(m.benchmark)} |")
    L.append("")

    rr = res.risk_return
    L += ["## Volatility vs return", "", "| series | n | return/yr | volatility/yr | Sharpe |", "|---|---|---|---|---|"]
    for _, r in rr.iterrows():
        L.append(f"| {r['name']} | {int(r.n)} | {_pct(r.ret)} | {_pct(r.vol, 1, )} | {'' if pd.isna(r.sharpe) else f'{r.sharpe:.2f}'} |")
    L.append("")
    rby = res.risk_by_year
    if rby.attrs.get("within_year_observable"):
        L += ["## Risk by year (from each year's monthly returns)", "",
              "VaR is the 95% one-month loss threshold: historical = the 5th percentile of that year's 12 months "
              "(so it sits between the worst and second-worst month — a coarse estimate), parametric = mean − 1.645σ. "
              "Expected shortfall = average of the months at or below the historical VaR.", "",
              "| year | return | vol/yr | variance/yr | VaR 95% hist | VaR 95% param | ES 95% | worst month | bench return | bench vol | bench VaR hist |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for y, r in rby.iterrows():
            L.append(f"| {y} | {_pct(r.return_p)} | {_pct(r.vol_p, 1)} | {_pct(r.variance_p)} | {_pct(r.var95_hist_p)} | "
                     f"{_pct(r.var95_param_p)} | {_pct(r.es95_p)} | {_pct(r.worst_p)} | {_pct(r.return_b)} | {_pct(r.vol_b, 1)} | {_pct(r.var95_hist_b)} |")
    else:
        tw = rby.attrs.get("trailing")
        L += [f"## Risk by year (trailing {tw}-year window; within-year figures are not observable from annual statements)", "",
              "| year | return | trailing vol/yr | trailing VaR 95% (one year) | trailing ES 95% | bench return | bench trailing vol | bench trailing VaR |",
              "|---|---|---|---|---|---|---|---|"]
        for y, r in rby.iterrows():
            L.append(f"| {y} | {_pct(r.return_p)} | {_pct(r.trailing_vol_p, 1)} | {_pct(r.trailing_var95_hist_p)} | {_pct(r.trailing_es95_p)} | "
                     f"{_pct(r.return_b)} | {_pct(r.trailing_vol_b, 1)} | {_pct(r.trailing_var95_hist_b)} |")
    L.append("")

    L += ["## Bootstrap", "", "| model | n | alpha/yr | t observed | p (null, one-sided) | p (two-sided) | "
          "bootstrap 95% CI for alpha | share of resamples with alpha ≤ 0 |", "|---|---|---|---|---|---|---|---|"]
    for _, b in res.bootstrap.iterrows():
        L.append(f"| {b.model} | {int(b.n)} | {_pct(b.alpha_annual)} | {b.t_observed:.2f} | {b.p_null_one_sided:.3f} | "
                 f"{b.p_null_two_sided:.3f} | {_pct(b.alpha_ci_low_annual, 1)} … {_pct(b.alpha_ci_high_annual, 1)} | "
                 f"{b.share_boot_alpha_below_zero:.1%} |")
    L += ["", f"{int(res.bootstrap.n_boot.iloc[0]):,} resamples of {unit}s with replacement. The null bootstrap "
          "imposes alpha = 0 and asks how often chance alone produces a t-statistic as large as the observed one.", ""]

    L += ["## Random-manager cohort", "", "| variant | managers | beta | resid. vol/yr | actual excess/yr | "
          "cohort p5 | p50 | p95 | percentile (excess) | percentile (t-alpha) | zero-skill managers ≥ actual |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, c in res.cohort.iterrows():
        L.append(f"| {c.variant} | {int(c.n_managers):,} | {c.beta:.2f} | {_pct(c.resid_sd_annual, 1)} | "
                 f"**{_pct(c.actual_headline_excess)}** | {_pct(c.cohort_excess_p5, 1)} | {_pct(c.cohort_excess_p50, 1)} | "
                 f"{_pct(c.cohort_excess_p95, 1)} | **{c.percentile_by_headline_excess:.0f}** | "
                 f"{c.percentile_by_t_alpha:.0f} | {c.share_of_zero_skill_managers_beating_actual:.1%} |")
    L += ["", "Each simulated manager has zero true skill, the record's own beta, and residuals bootstrapped "
          "from the record's own residuals. 'Same market path' answers: given these exact markets, how lucky "
          "would a no-skill manager have to be? 'Resampled' also randomizes which market years occurred.", ""]

    L += ["## Sub-periods", "", "| model | segment | n | alpha/yr | SE | t | p |", "|---|---|---|---|---|---|---|"]
    for _, s in res.subperiods.iterrows():
        L.append(f"| {s.model} | {s.segment} | {int(s.n)} | {_pct(s.alpha_annual)}{_stars(s.p)} | {_pct(s.se_annual)} | {s.t:.2f} | {s.p:.3f} |")
    L += ["", f"Split at {cfg.split_year}. 'difference' tests whether alpha changed; a small p there means the "
          "two halves are genuinely different, not just noisy.", ""]

    roll = res.rolling
    dcol = [c for c in roll.columns if c.startswith("excess_")]
    acol = [c for c in roll.columns if c.startswith("alpha_") and c.count("_") == 1]
    L += ["## Rolling windows", ""]
    if dcol and acol:
        wd, wa = dcol[0].split("_")[1], acol[0].split("_")[1]
        L += [f"| {unit} ending | {wd}-{unit} rolling excess/yr | {wa}-{unit} rolling alpha/yr | 95% CI | t | beta |",
              "|---|---|---|---|---|---|"]
        for cell, r in roll.iterrows():
            e = r.get(dcol[0], np.nan); a = r.get(acol[0], np.nan)
            if pd.isna(e) and pd.isna(a):
                continue
            L.append(f"| {cell} | {_pct(e)} | {_pct(a)} | "
                     + (f"{_pct(r[acol[0] + '_ci_low'], 1)} … {_pct(r[acol[0] + '_ci_high'], 1)} | {r[acol[0] + '_t']:.2f} | {r['beta_' + wa]:.2f} |"
                        if not pd.isna(a) else "| | |"))
        pos = roll[dcol[0]].dropna()
        L += ["", f"Rolling {wd}-{unit} excess return was positive in {int((pos > 0).sum())} of {len(pos)} windows; "
              f"lowest {_pct(pos.min())}, highest {_pct(pos.max())}."]
        al = roll[acol[0]].dropna()
        if len(al):
            sig = roll.loc[al.index, acol[0] + "_ci_low"] > 0
            L.append(f"Rolling {wa}-{unit} alpha was positive in {int((al > 0).sum())} of {len(al)} windows and "
                     f"its 95% CI excluded zero in {int(sig.sum())}.")
    L.append("")
    L += ["## Caveats", ""] + [f"- {n}" for n in res.notes] + [
        "- All regressions are in-sample over the full record. There is no out-of-sample test possible "
        "for a single manager's history; the sub-period split is the nearest substitute.",
        "- The composite is gross of fees (none were charged). Alpha net of any proposed fee is "
        "alpha minus the fee drag shown in Phase 2.", ""]
    (out / "summary.md").write_text("\n".join(L))
    return out / "summary.md"
