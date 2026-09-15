"""Due-diligence memo for one manager, assembled from the pipeline's own outputs.

The memo answers, in the order an investment committee asks them: what is claimed,
what can be verified, is the excess return skill or exposure, is it statistically
believable, is it stable, what are the risks, and what should we ask the manager.
Every sentence is generated from numbers in output/<fund>/ (phases 2–4) and
data/funds/<slug>/ (filings, contributions), so a rebuild cannot leave stale prose.

The verdict rule is fixed and the same for every manager:
  evidence     FF3 alpha t ≥ 2, bootstrap p < 0.05 and fewer than 10% of zero-skill managers do as well
  suggestive   1 ≤ t < 2
  none         |t| < 1
  negative     t ≤ −1
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .dashboard import CSS, JS, esc, line_chart, pct, num

ROOT = Path(__file__).resolve().parents[1]
DRIFT = 0.30            # a factor loading that moves this much between the first and last window is "drift"
WINDOW = 60             # months per rolling window (36 when the record is short)


def _read(p: Path, **kw) -> pd.DataFrame | None:
    return pd.read_csv(p, **kw) if p.exists() else None


def rolling_loadings(al: pd.DataFrame, window: int) -> pd.DataFrame:
    """Rolling FF3 loadings and alpha from the aligned monthly data (y = r_p − RF)."""
    cols = ["MKT_RF", "SMB", "HML"]
    d = al.dropna(subset=["y"] + cols)
    rows = []
    for i in range(window, len(d) + 1):
        s = d.iloc[i - window:i]
        X = sm.add_constant(s[cols].values)
        f = sm.OLS(s.y.values, X).fit()
        rows.append(dict(cell=s.cell.iloc[-1], alpha=f.params[0] * 12, MKT=f.params[1], SMB=f.params[2], HML=f.params[3], r2=f.rsquared))
    return pd.DataFrame(rows)


def split_alpha(al: pd.DataFrame) -> tuple[dict, dict]:
    cols = ["MKT_RF", "SMB", "HML"]
    d = al.dropna(subset=["y"] + cols)
    h = len(d) // 2
    out = []
    for s in (d.iloc[:h], d.iloc[h:]):
        f = sm.OLS(s.y.values, sm.add_constant(s[cols].values)).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
        out.append(dict(first=s.cell.iloc[0], last=s.cell.iloc[-1], n=len(s), alpha=f.params[0] * 12, t=f.tvalues[0]))
    return out[0], out[1]


def drawdown_episode(r: pd.Series) -> dict:
    cum = (1 + r).cumprod(); peak = cum.cummax(); dd = cum / peak - 1
    trough = dd.idxmin(); start = cum.loc[:trough].idxmax()
    rec = cum.loc[trough:]; back = rec[rec >= peak.loc[trough]]
    return dict(depth=float(dd.min()), start=str(start), trough=str(trough), recovered=str(back.index[0]) if len(back) else None,
                months_down=int(r.loc[start:trough].shape[0] - 1))


def verdict(t: float, p_boot: float | None, cohort_share: float | None) -> tuple[str, str]:
    if pd.isna(t):
        return "unknown", "not testable"
    if t >= 2 and (p_boot is None or p_boot < 0.05) and (cohort_share is None or cohort_share < 0.10):
        return "yes", "evidence of skill"
    if t >= 1:
        return "weak", "suggestive, not established"
    if t > -1:
        return "no", "no evidence of skill"
    return "neg", "evidence of negative alpha"


def build_memo(out_dir: Path, data_dir: Path, label: str | None = None) -> str | None:
    p4 = out_dir / "phase4"
    regs = _read(p4 / "regressions.csv"); metrics = _read(p4 / "metrics.csv"); scores = _read(p4 / "scores.csv")
    boot = _read(p4 / "bootstrap.csv"); coh = _read(p4 / "cohort.csv"); al = _read(p4 / "aligned_data.csv"); skill = _read(p4 / "skill.csv")
    comp = _read(out_dir / "phase2" / "composite_returns.csv"); timing = _read(out_dir / "phase3" / "timing_tests.csv")
    if regs is None or al is None or comp is None:
        return None
    meta = json.loads((data_dir / "meta.json").read_text()) if (data_dir / "meta.json").exists() else {}
    filings = _read(data_dir / "filings.csv"); contrib = _read(data_dir / "contributions.csv")
    name = label or meta.get("name") or out_dir.name
    manager = meta.get("manager", ""); style_name = meta.get("style_name", ""); style_note = meta.get("style_note", "")
    is_clone = "13F" in (meta.get("style_note", "") + str(comp.get("custodian", ""))) or (data_dir / "filings.csv").exists()

    us = regs[regs.factor_set == "US"].set_index("model")
    ff3, capm, car, ff5 = us.loc["FF3"], us.loc["CAPM"], us.loc["Carhart4"], (us.loc["FF5"] if "FF5" in us.index else None)
    m = metrics.set_index("metric")
    def mv(k, col="portfolio"):
        try: return float(m.loc[k, col])
        except Exception: return np.nan
    ann_p, ann_b = mv("annualized return"), mv("annualized return", "benchmark")
    vol_p, vol_b = mv("annualized volatility"), mv("annualized volatility", "benchmark")
    dd_p, dd_b = mv("max drawdown (cell-level; understated on annual cells)"), mv("max drawdown (cell-level; understated on annual cells)", "benchmark")
    sharpe_p, sharpe_b = mv("Sharpe (excess over RF)"), mv("Sharpe (excess over RF)", "benchmark")
    te, ir = mv("tracking error (ann.)"), mv("information ratio")
    down_cap, up_cap = mv("down capture"), mv("up capture")
    es_p, es_b = mv("expected shortfall 95%, one month"), mv("expected shortfall 95%, one month", "benchmark")
    worst = mv("worst month")
    p_boot = float(boot[boot.model == "FF3"].p_null_two_sided.iloc[0]) if boot is not None and (boot.model == "FF3").any() else None
    coh_row = coh[(coh.model == "FF3") & (coh.variant == "same market path")] if coh is not None else None
    coh_share = float(coh_row.share_of_zero_skill_managers_beating_actual.iloc[0]) if coh_row is not None and len(coh_row) else None
    sc = scores.set_index(["score", "component"]) if scores is not None else None
    def score(s, c):
        try: return float(sc.loc[(s, c), "value"])
        except Exception: return np.nan
    am, wm = score("Alpha-maxing score", "excess return over market"), score("Wealth-management score", "TOTAL")
    consistency = score("Wealth-management score", "consistency")
    v_cls, v_lab = verdict(float(ff3.t), p_boot, coh_share)

    # stability
    win = WINDOW if len(al.dropna(subset=["y"])) >= WINDOW + 24 else 36
    roll = rolling_loadings(al, win)
    drift = {}
    if len(roll) >= 2:
        f0, f1 = roll.iloc[0], roll.iloc[-1]
        for k in ["MKT", "SMB", "HML"]:
            drift[k] = dict(first=float(f0[k]), last=float(f1[k]), w0=f0.cell, w1=f1.cell, moved=abs(float(f1[k]) - float(f0[k])) >= DRIFT)
    h1, h2 = split_alpha(al)
    r = comp.set_index("cell").gross.astype(float); r.index = r.index.astype(str)
    ddep = drawdown_episode(r)
    # concentration
    conc = None
    if contrib is not None and len(contrib):
        pos = contrib[contrib.active > 0]
        n50 = int((pos.cum_share_of_outperformance < 0.5).sum() + 1) if len(pos) else 0
        top5 = float(pos.share_of_outperformance.head(5).sum()) if len(pos) else np.nan
        conc = dict(n50=n50, top5=top5, n_all=int(len(contrib)), n_win=int(len(pos)), top=pos.head(5)[["ticker", "name", "active", "months_held"]].to_dict("records"))
    fil = None
    if filings is not None and len(filings):
        fil = dict(n_last=int(filings.n.iloc[-1]), n_med=float(filings.n.median()), cov_last=float(filings.priced_share.iloc[-1]), cov_avg=float(filings.priced_share.mean()),
                   last_period=str(filings.period.iloc[-1]), first_period=str(filings.period.iloc[0]), n_filings=int(len(filings)))
    tm = timing[timing.model == "Treynor-Mazuy"].iloc[0] if timing is not None and (timing.model == "Treynor-Mazuy").any() else None
    # model-net vs benchmark
    net_ann = None
    if "model_net" in comp and comp.model_net.notna().any():
        mn = comp.model_net.astype(float).dropna(); net_ann = float((1 + mn).prod() ** (12 / len(mn)) - 1)
    first, last = str(comp.cell.iloc[0]), str(comp.cell.iloc[-1]); months = int(len(comp))
    years = months / 12

    # ---------------------------------------------------------------- questions for the manager
    qs = []
    for k, lab in [("MKT", "market beta"), ("SMB", "size (small-cap) exposure"), ("HML", "value exposure")]:
        d = drift.get(k)
        if d and d["moved"]:
            qs.append(f"Your {lab} moved from {d['first']:+.2f} in the window ending {d['w0']} to {d['last']:+.2f} in the window ending {d['w1']}. "
                      f"Was that a deliberate change in the process, or drift from what worked?")
    if abs(h2["alpha"] - h1["alpha"]) >= 0.03:
        qs.append(f"FF3 alpha was {pct(h1['alpha'], 1)}/yr in {h1['first']}–{h1['last']} and {pct(h2['alpha'], 1)}/yr in {h2['first']}–{h2['last']}. "
                  + ("What explains the decay — capacity, crowding of the ideas, or a regime the process is not built for?" if h2["alpha"] < h1["alpha"]
                     else "What changed to produce the improvement, and is it repeatable?"))
    if fil:
        if fil["n_last"] < 0.6 * fil["n_med"] or fil["n_last"] > 1.5 * fil["n_med"]:
            qs.append(f"The disclosed book holds {fil['n_last']} names in the latest filing against a typical {fil['n_med']:.0f}. "
                      f"Is the change in concentration a view, flows, or a change in the mandate?")
        if fil["cov_avg"] < 0.75:
            qs.append(f"Only {fil['cov_avg']:.0%} of disclosed value maps to priced US equities on average. What is in the remainder — "
                      f"foreign listings, unlisted, options — and how is it risk-managed?")
    if conc and (conc["top5"] >= 0.6 or conc["n50"] <= 3):
        tops = ", ".join(f"{t['ticker']}" for t in conc["top"][:3])
        qs.append(f"Half of the outperformance came from {conc['n50']} name{'s' if conc['n50'] != 1 else ''} ({tops}…), the top five for {conc['top5']:.0%}. "
                  f"What is the repeatable process behind those, and does it scale beyond a handful of positions?")
    qs.append(f"The book fell {pct(ddep['depth'], 0, False)} from {ddep['start']} to {ddep['trough']}"
              + (f" and recovered by {ddep['recovered']}" if ddep["recovered"] else " and has not yet recovered")
              + ". What was the risk-management response, and what would trigger de-risking today?")
    if tm is not None and float(tm.gamma_p) < 0.10:
        qs.append(f"The Treynor–Mazuy test finds {'convex' if float(tm.gamma) > 0 else 'concave'} market exposure (γ = {float(tm.gamma):+.2f}, p = {float(tm.gamma_p):.2f}). "
                  f"Is beta being managed explicitly, or is it a by-product of the stock selection?")
    if net_ann is not None and ann_p > ann_b > net_ann:
        qs.append(f"Gross of fees the record beat its benchmark by {pct(ann_p - ann_b, 1)}/yr; at a 0.5% and 20% fee a client would have trailed it by {pct(ann_b - net_ann, 1)}/yr. "
                  f"How is the fee level justified against a passive alternative with the same factor exposures?")
    if is_clone:
        qs.append("The analysis sees only the disclosed long US book, 45 days late. What share of gross exposure is short, non-US, derivative or cash, "
                  "and how has the long book's return compared with the fund's actual net return over the same period?")
    if meta.get("dropped_months"):
        qs.append(f"{meta['dropped_months']} months were dropped because too little of the book could be priced. Can you supply the full position-level history so the gaps close?")
    if coh_share is not None and coh_share > 0.2:
        qs.append(f"{coh_share:.0%} of simulated managers with no skill and the same factor exposures matched this record. What evidence, beyond the return series, "
                  f"distinguishes the process from luck — hit rates, attribution by thesis, out-of-sample tests?")

    # ---------------------------------------------------------------- rendering helpers
    def load_row(mdl, lab):
        r_ = us.loc[mdl]
        cells = "".join(f"<td class='n'>{num(r_.get(f'b_{f}'))}</td>" if not pd.isna(r_.get(f"b_{f}", np.nan)) else "<td class='n'>—</td>" for f in ["MKT_RF", "SMB", "HML", "MOM", "RMW", "CMA"])
        return f"<tr><td><b>{lab}</b></td><td class='n'>{pct(r_.alpha_annual, 1)}</td><td class='n'>{pct(r_.ci_low_annual, 1)} … {pct(r_.ci_high_annual, 1)}</td><td class='n'>{r_.t:+.2f}</td><td class='n'>{r_.p:.3f}</td>{cells}<td class='n'>{r_.r2:.2f}</td></tr>"
    loads = load_row("CAPM", "CAPM") + load_row("FF3", "Fama–French 3") + load_row("Carhart4", "Carhart 4") + (load_row("FF5", "Fama–French 5") if ff5 is not None else "")
    def expo(k, v):
        if pd.isna(v): return ""
        if k == "MKT_RF": return f"β of {v:.2f}: {'more' if v > 1.05 else 'less' if v < 0.95 else 'about the same'} market risk as the index."
        if k == "SMB": return f"{'a tilt to smaller companies' if v > 0.15 else 'a tilt to the largest companies' if v < -0.15 else 'no size tilt'} ({v:+.2f})."
        if k == "HML": return f"{'a value tilt — cheap stocks' if v > 0.15 else 'a growth tilt — expensive stocks' if v < -0.15 else 'neither value nor growth'} ({v:+.2f})."
        if k == "MOM": return f"{'rides momentum' if v > 0.15 else 'leans against momentum' if v < -0.15 else 'momentum-neutral'} ({v:+.2f})."
    expo_txt = " ".join(e for e in [expo("MKT_RF", ff3.b_MKT_RF), expo("SMB", ff3.b_SMB), expo("HML", ff3.b_HML), expo("MOM", car.get("b_MOM", np.nan))] if e)
    # rolling chart
    roll_chart = ""
    if len(roll) >= 6:
        cells = list(roll.cell)
        roll_chart = line_chart(cells, [dict(name="market β", values=[float(v) for v in roll.MKT], cls="s0"), dict(name="size SMB", values=[float(v) for v in roll.SMB], cls="s1"),
                                        dict(name="value HML", values=[float(v) for v in roll.HML], cls="s1l")], height=240, width=860, y_fmt=lambda v: f"{v:+.1f}", uid="roll")
    drift_txt = "; ".join(f"{k} {d['first']:+.2f} → {d['last']:+.2f}" + (" <b>(drift)</b>" if d["moved"] else "") for k, d in drift.items()) if drift else "record too short for rolling windows"
    conc_txt = (f"Half of the outperformance over the market came from <b>{conc['n50']}</b> of {conc['n_all']} names ever held; the top five account for {conc['top5']:.0%}. "
                + "Largest contributors: " + ", ".join(f"{t['ticker']} ({t['active'] * 100:+.0f} pp over {t['months_held']} mo)" for t in conc["top"][:5]) + ".") if conc else "Position-level contributions are not available for this record."
    q_html = "".join(f"<li>{esc(q)}</li>" for q in qs)
    chip = f'<span class="chip {"verified" if v_cls == "yes" else "estimated" if v_cls == "weak" else "unverified"}">{esc(v_lab)}</span>'
    bench_name = "US market (Ken French total market)"
    rep_te = float(ff3.resid_sd_annual)
    net_line = (f" A client paying 0.5% and 20% above a high-water mark would have kept {pct(net_ann, 1, False)}/yr — {'ahead of' if net_ann > ann_b else 'behind'} the index." if net_ann is not None else "")
    summary = (f"{esc(name)}{' (' + esc(manager) + ')' if manager else ''}, {esc(style_name.lower()) if style_name else 'the record'}, "
               f"{'as seen through its disclosed US long book — a 13F clone, not the fund — ' if is_clone else ''}returned <b>{pct(ann_p, 1, False)}</b> a year over {first}–{last} "
               f"against <b>{pct(ann_b, 1, False)}</b> for the market, an excess of {pct(ann_p - ann_b, 1)}. After removing what market, size and value exposure explain, the alpha is "
               f"<b>{pct(ff3.alpha_annual, 1)}</b>/yr with a 95% range of {pct(ff3.ci_low_annual, 1)} to {pct(ff3.ci_high_annual, 1)} (t = {ff3.t:+.2f}"
               + (f", bootstrap p = {p_boot:.2f}" if p_boot is not None else "") + ")"
               + (f"; {coh_share:.0%} of simulated zero-skill managers with the same exposures did as well" if coh_share is not None else "") + ". "
               f"Verdict: <b>{v_lab}</b>.{net_line}")
    claimed = ("The return series is reconstructed from SEC 13F-HR filings and Yahoo Finance prices: the disclosed US long positions at disclosed weights, "
               "bought at the end of the month each filing became public and held with drift until the next. Nothing in it is the fund's own statement of "
               "returns, so every period is marked <i>unverified</i> by the pipeline — the correct label for a reconstruction. " if is_clone else
               "Returns are computed from the source statements listed in the coverage report; periods carry the evidence tier the reconciliation assigned. ")
    clone_caveat = ("The clone sees the long US book only, 45 days late; the fund's actual net return may differ materially, in either direction. "
                    "Request the fund's audited series before any decision." if is_clone else "")
    fil_txt = (f"{fil['n_filings']} filings from {fil['first_period']} to {fil['last_period']}; {fil['cov_avg']:.0%} of disclosed value priced on average "
               f"({fil['cov_last']:.0%} in the latest), {fil['n_med']:.0f} names in a typical filing. " if fil else "")
    drop_txt = f"{meta['dropped_months']} months dropped for insufficient priced coverage; " if meta.get("dropped_months") else ""
    yrs_needed = max(1, int(round((2 * ff3.se_annual / ff3.alpha_annual) ** 2 * years))) if ff3.alpha_annual > 0 else None
    believe = (f"The parametric test gives p = {ff3.p:.3f}"
               + (f"; resampling the record with its alpha removed produces a t this large {p_boot:.0%} of the time" if p_boot is not None else "")
               + (f"; of {int(coh_row.n_managers.iloc[0]):,} simulated managers with <b>no skill</b> and the same FF3 loadings and residual risk, {coh_share:.0%} matched or beat the record" if coh_share is not None else "")
               + f". Over {years:.1f} years the alpha's standard error is {pct(ff3.se_annual, 1, False)}/yr; "
               + (f"to establish an alpha of this size at t = 2 would take roughly {yrs_needed if yrs_needed <= 100 else 'more than a hundred'} years of the same performance. " if yrs_needed else "a negative point estimate cannot be established as skill by waiting. ")
               + "Across the 89 scorable managers in the universe, about 4–5 would show t ≥ 2 by chance alone at the 5% level — a single manager's t must be read against that.")
    stab = (f"FF3 alpha by half: <b>{pct(h1['alpha'], 1)}</b>/yr (t {h1['t']:+.1f}) over {h1['first']}–{h1['last']} versus <b>{pct(h2['alpha'], 1)}</b>/yr (t {h2['t']:+.1f}) over {h2['first']}–{h2['last']}. "
            f"Beat the market in {consistency:.0f}% of rolling five-year windows."
            + (f" Market timing (Treynor–Mazuy): γ = {float(tm.gamma):+.2f}, p = {float(tm.gamma_p):.2f} — {'some evidence of timing' if float(tm.gamma_p) < 0.1 else 'no evidence of timing'}." if tm is not None else ""))
    return f"""<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(name)} — due-diligence memo</title>
<style>{CSS}
.memo h3 {{ margin-top: 0 }} .memo ol.q li {{ margin: 8px 0 }} .memo .chip {{ vertical-align: middle }}
.kv {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px }} .kv .tile .tv {{ font-size: 26px }}
@media print {{ .tr-nav, .banner, details.how {{ display: none !important }} .cover {{ -webkit-print-color-adjust: exact; print-color-adjust: exact }} section {{ break-inside: avoid }} }}
</style>
<div class="banner" role="note"><span class="bl">{'Illustrative data' if is_clone else 'Memo'}</span> {'A 13F long-only clone of the disclosed US book — a reconstruction, not the fund. ' if is_clone else ''}Generated from the pipeline's outputs; every figure links back to a table on the dashboard.</div>
<header class="cover"><div class="cover-in">
  <div class="cover-top"><div class="eyebrow">Investment due diligence · memorandum</div></div>
  <div class="gold-rule"></div>
  <h1>{esc(name)}</h1>
  <p class="sub">{esc(manager)}{' · ' if manager and style_name else ''}{esc(style_name)}{' — ' + esc(style_note) if style_note else ''}</p>
  <dl class="meta">
    <div><dt>Period</dt><dd>{esc(first)} – {esc(last)}</dd></div>
    <div><dt>Observations</dt><dd>{months} months</dd></div>
    <div><dt>Benchmark</dt><dd>{esc(bench_name)}</dd></div>
    <div><dt>Prepared</dt><dd>{date.today().strftime('%d %B %Y')}</dd></div>
    <div><dt>Verdict</dt><dd>{esc(v_lab)}</dd></div>
  </dl>
</div></header>
<main class="wrap memo">
<section class="verdict">
  <div class="sh"><h2>Summary</h2>{chip}</div>
  <div class="card"><p style="font-size:16px">{summary}</p>
  <div class="kv" style="margin-top:12px">
    <div class="tile"><div class="tl">Return /yr</div><div class="tv">{pct(ann_p, 1, False)}</div><div class="td muted">market {pct(ann_b, 1, False)}</div></div>
    <div class="tile"><div class="tl">FF3 alpha /yr</div><div class="tv">{pct(ff3.alpha_annual, 1)}</div><div class="td muted">t {ff3.t:+.2f} · p {ff3.p:.3f}</div></div>
    <div class="tile"><div class="tl">Information ratio</div><div class="tv">{num(ir)}</div><div class="td muted">tracking error {pct(te, 1, False)}</div></div>
    <div class="tile"><div class="tl">Scores</div><div class="tv">{am:.0f} · {wm:.0f}</div><div class="td muted">alpha-maxing · wealth-management, of 100</div></div>
  </div></div>
</section>

<section>
  <div class="sh"><h2>What is claimed and what is verified</h2></div>
  <div class="card"><p>{claimed}{fil_txt}{drop_txt}Time-weighted, gross; no fees modelled unless stated.</p></div>
</section>

<section>
  <div class="sh"><h2>Skill or exposure?</h2></div>
  <div class="card tscroll"><table><thead><tr><th>model</th><th class="n">alpha /yr</th><th class="n">95% range</th><th class="n">t</th><th class="n">p</th><th class="n">Mkt</th><th class="n">SMB</th><th class="n">HML</th><th class="n">MOM</th><th class="n">RMW</th><th class="n">CMA</th><th class="n">R²</th></tr></thead><tbody>{loads}</tbody></table>
  <p class="cap">Each row regresses the monthly return over T-bills on factor premia anyone can buy cheaply; the intercept is what is left for the manager. Newey–West standard errors.</p>
  <p><b>Reading the exposures.</b> {expo_txt} Jensen's (CAPM) alpha of {pct(capm.alpha_annual, 1)} {'exceeds' if capm.alpha_annual > ff3.alpha_annual else 'is below'} the FF3 alpha, so {'part of the raw outperformance is a size/value tilt an index fund could have bought' if capm.alpha_annual > ff3.alpha_annual else 'the size/value tilts worked against the record — the FF3 alpha is the fairer number'}.</p>
  <p><b>Replication test.</b> A three-factor portfolio explains {ff3.r2:.0%} of the month-to-month variance; what it cannot replicate is {pct(rep_te, 1, False)}/yr of tracking error carrying an alpha of {pct(ff3.alpha_annual, 1)} — an appraisal ratio of {num(ff3.alpha_annual / rep_te if rep_te else np.nan)}. {'A passive factor sleeve at a few basis points would have delivered most of this record.' if ff3.r2 >= 0.85 and ff3.t < 2 else 'The unexplained part is large enough that the record is not simply factor exposure in disguise.' if ff3.r2 < 0.7 else 'The record is mostly, but not entirely, replicable with factor exposures.'}</p></div>
</section>

<section>
  <div class="sh"><h2>Is it believable?</h2></div>
  <div class="card"><p>{believe}</p></div>
</section>

<section>
  <div class="sh"><h2>Stability and drift</h2></div>
  <div class="card">{roll_chart}<p class="cap">Rolling {win}-month Fama–French loadings. {drift_txt}.</p>
  <p>{stab}</p></div>
</section>

<section>
  <div class="sh"><h2>Risk</h2></div>
  <div class="card"><div class="kv">
    <div class="tile"><div class="tl">Volatility</div><div class="tv">{pct(vol_p, 1, False)}</div><div class="td muted">market {pct(vol_b, 1, False)} · β {num(capm.b_MKT_RF)}</div></div>
    <div class="tile"><div class="tl">Max drawdown</div><div class="tv">{pct(dd_p, 0, False)}</div><div class="td muted">market {pct(dd_b, 0, False)} · {ddep['start']} → {ddep['trough']}</div></div>
    <div class="tile"><div class="tl">Down / up capture</div><div class="tv">{num(down_cap)} / {num(up_cap)}</div><div class="td muted">{'loses more than the market in down months' if down_cap > 1.05 else 'cushions down months' if down_cap < 0.95 else 'tracks the market on the downside'}</div></div>
    <div class="tile"><div class="tl">Expected shortfall 95%</div><div class="tv">{pct(es_p, 1, False)}</div><div class="td muted">market {pct(es_b, 1, False)} · worst month {pct(worst, 1)}</div></div>
    <div class="tile"><div class="tl">Sharpe</div><div class="tv">{num(sharpe_p)}</div><div class="td muted">market {num(sharpe_b)}</div></div>
  </div>
  <p style="margin-top:12px"><b>Concentration.</b> {conc_txt}</p></div>
</section>

<section>
  <div class="sh"><h2>Questions for the manager</h2></div>
  <div class="card"><ol class="q">{q_html}</ol>
  <p class="cap">Generated from the record: each question points at a number above. Bring the dashboard to the meeting.</p></div>
</section>

<section>
  <div class="sh"><h2>Recommendation</h2></div>
  <div class="card"><p>{
      'The record shows alpha that survives factor adjustment and simulation. Proceed to operational due diligence; the questions above are the agenda for the first meeting, with particular attention to capacity and stability of the exposures.' if v_cls == 'yes' else
      'The record is consistent with modest skill but does not establish it. A mandate is defensible only if the process, not the track record, is the case — the meeting should test that. A passive factor portfolio with the same exposures is the alternative to beat.' if v_cls == 'weak' else
      'The excess return, where there is one, is explained by exposures that can be bought passively at low cost; there is no statistical evidence of selection skill. Do not proceed on the return record; revisit only with a process-based case or a materially longer history.' if v_cls == 'no' else
      'The record shows negative alpha after factor adjustment. Do not proceed.'}</p>
  <p class="cap">{clone_caveat}</p></div>
</section>

<footer class="foot">
  <div class="running"><span>Due-diligence memorandum · {esc(name)}</span><span>{esc(first)} – {esc(last)}</span></div>
  <h4>Important information</h4>
  <p>Assembled automatically from the verification pipeline (COMPOSITE_RULES.md; phases 2–4). {'Built on public SEC EDGAR filings and Yahoo Finance prices; a reconstruction of the disclosed long book, not the fund. ' if is_clone else ''}Factor data: Kenneth R. French Data Library. Statistical results are in-sample; the confidence interval, not the point estimate, is the claim. Not investment advice.</p>
</footer>
</main>
<div id="tip" class="tip" hidden></div>
<script>{JS}</script>
"""
