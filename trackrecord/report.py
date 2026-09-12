"""Top-level REPORT.md: what is verified, what is estimated, what is unknown,
and how the verified number reconciles to the claimed one."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .composite import series_stats
from .returns import chain, annualize


def _pct(x, d=2):
    return "n/a" if x is None or pd.isna(x) else f"{x * 100:+.{d}f}%"


def reconciliation_rows(res2, comp, acc, g, n, pri, surv) -> list[dict]:
    """Candidate calculations a self-computed figure might rest on, from the same data."""
    rows = [dict(calculation="Verified composite TWR (the right number)", value=g["annualized"], why="—", is_verified=True)]
    if pri:
        rows.append(dict(calculation="Principal's account only, TWR", value=pri["annualized"], why="one account, not the client record", is_verified=False))
    p_acc = acc[acc.owner_type == "principal"]
    if len(p_acc):
        pa = p_acc.iloc[0]
        rows.append(dict(calculation="Principal's account, naive CAGR (deposits counted as gains)", value=pa.naive_cagr_ignoring_flows, why="contributions inflate ending value", is_verified=False))
        rows.append(dict(calculation="Principal's account, average of yearly returns", value=pa.arithmetic_mean_annual, why="a plain average ignores compounding / volatility drag", is_verified=False))
    rows.append(dict(calculation="Composite, average of yearly returns", value=g["arithmetic_mean_annual"], why="a plain average ignores volatility drag", is_verified=False))
    if surv:
        rows.append(dict(calculation="Survivors-only composite", value=surv["annualized"], why="drops closed accounts", is_verified=False))
    best = acc.dropna(subset=["twr_annualized"]).sort_values("twr_annualized", ascending=False)
    if len(best):
        bb = best.iloc[0]
        span = f"{str(bb.twr_span)[:4]}–{str(bb.twr_span)[-10:-6]}" if ".." in str(bb.twr_span) else str(bb.twr_span)
        rows.append(dict(calculation=f"Best single account ({bb.account_id}, {span})", value=bb.twr_annualized, why=f"cherry-picked account and span ({bb.twr_span})", is_verified=False))
    tw = comp.tail(10)
    if len(tw) == 10:
        rows.append(dict(calculation="Last 10 years only", value=annualize(chain(tw.gross), float(tw.years.sum())), why="favorable start date", is_verified=False))
    rows.append(dict(calculation="Model-net at assumed fees", value=n["annualized"], why="what a paying client would have received", is_verified=False))
    return rows


BENCH_NAMES = {"US_MKT": "Ken French US market total return", "DEV_MKT": "Ken French developed-market total return",
               "DXUS_MKT": "Ken French developed ex-US total return"}


def write_report(res1, res2, res3, res4, out_dir, data_dir, claimed: float | None = None,
                 placeholder: bool = False, placeholder_note: str = "placeholder data",
                 claimed_note: str = "") -> Path:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    st = res1.statements; comp = res2.composite; acc = res2.accounts; cfg = res2.config
    g = series_stats(comp.gross, comp.years); n = series_stats(comp.model_net, comp.years)
    b = series_stats(comp.benchmark, comp.years); surv = series_stats(comp.survivors_only, comp.years)
    pri = series_stats(comp.principal_only, comp.years); cli = series_stats(comp.clients_only, comp.years)
    R = res4.regressions.set_index(["factor_set", "model"])
    hm = res4.config.headline_model
    capm = R.loc[(res4.config.factor_set, hm)]          # headline model (FF3 by default)
    jens = R.loc[(res4.config.factor_set, "CAPM")]      # Jensen's alpha = CAPM intercept
    boot = res4.bootstrap.set_index("model").loc[hm]
    coh = res4.cohort.iloc[0]
    s1 = res1.summary()

    L = ["# Track-record verification report", ""]
    monthly = cfg.grid == "M"
    unit = "account-months" if monthly else "account-years"
    if placeholder:
        L += [f"> **PLACEHOLDER DATA.** Every number below comes from {placeholder_note}. None of it describes "
              "the record under verification.", ""]
    L += [f"Generated {date.today()} from `{data_dir}`. Composite rules: COMPOSITE_RULES.md. "
          "Phase detail: coverage.md, phase2/summary.md, phase3/summary.md, phase4/summary.md.", ""]

    # 1. the number
    bench_id = str(acc.benchmark.mode().iloc[0]) if len(acc) else cfg.default_benchmark
    L += ["## 1. The number", "",
          "| | annualized | period | basis |", "|---|---|---|---|",
          f"| **Composite, time-weighted, gross (= net; no fees charged)** | **{_pct(g['annualized'])}** | {g['first']}–{g['last']} | verified statements, all discretionary accounts incl. closed |",
          f"| Composite, model-net ({cfg.fee.describe()}) | {_pct(n['annualized'])} | same | **assumed** fee schedule |",
          f"| Benchmark ({bench_id}) | {_pct(b['annualized'])} | same | {BENCH_NAMES.get(bench_id, bench_id)} |",
          f"| Excess over benchmark, gross | {_pct(g['annualized'] - b['annualized'])}/yr | same | |",
          f"| {hm} alpha (95% CI) | {_pct(capm.alpha_annual)} ({_pct(capm.ci_low_annual, 1)} … {_pct(capm.ci_high_annual, 1)}) | {int(capm.n)} obs | t = {capm.t:.2f}, p = {capm.p:.3f}; market, size and value premia removed |",
          f"| Jensen's alpha (CAPM, 95% CI) | {_pct(jens.alpha_annual)} ({_pct(jens.ci_low_annual, 1)} … {_pct(jens.ci_high_annual, 1)}) | {int(jens.n)} obs | t = {jens.t:.2f}; market premium only |",
          f"| Chance a zero-skill manager matches this | {coh.share_of_zero_skill_managers_beating_actual:.1%} | | {int(coh.n_managers):,}-manager simulation, same {coh.model} loadings & residual risk |", ""]
    if claimed is not None:
        gap = g["annualized"] - claimed
        L += [f"Claimed: **{_pct(claimed)}** — the figure the manager states, taken as an input. "
              f"Verified composite: **{_pct(g['annualized'])}** — computed above from the statements. "
              f"Gap: **{_pct(gap)}/yr**." + (f" {claimed_note}" if claimed_note else "")
              + ("" if gap >= 0 else " See §3 for where the difference is likely to come from."), ""]

    # 2. verified / estimated / unknown
    tot_ay = int(comp.n.sum()); ver = int(comp.n_verified.sum()); unv = int(comp.n_unverified.sum())
    gaps = res1.statements  # gaps computed in coverage; recompute count of PERIOD_GAP flags
    n_gaps = int((res1.flags.code == "PERIOD_GAP").sum()) if len(res1.flags) else 0
    flagged_ids = sorted(st.loc[st.status == "flagged", "statement_id"])
    unverified_ids = sorted(st.loc[st.status == "unverified", "statement_id"])
    L += ["## 2. What is verified, estimated, and unknown", "",
          "### Verified", "",
          f"- {s1['statements']} statement periods across {s1['accounts']} accounts were parsed; "
          f"**{s1['verified']}** passed at least one independent reconciliation check with none failing.",
          f"- The composite rests on **{tot_ay} {unit}**, of which **{ver} ({ver / tot_ay:.0%})** are verified "
          f"and {unv} are unverified (a value exists, nothing on the page confirms it).",
          f"- Closed/terminated accounts included: {int((acc.status != 'open').sum())} of {len(acc)}. "
          f"Survivor-only composite would show {_pct(surv.get('annualized'))} vs {_pct(g['annualized'])} "
          f"(effect {_pct(surv.get('annualized', np.nan) - g['annualized'])}/yr).",
          f"- Principal's own account: {_pct(pri.get('annualized'))} ({pri.get('first')}–{pri.get('last')}"
          + (" ⚠ truncated by a hole" if pri.get("truncated") else "") + f"); clients only: {_pct(cli.get('annualized'))}.",
          f"- Money-weighted (pooled IRR): {_pct(float(res2.irr[res2.irr.account_id == 'POOLED_DISCRETIONARY'].irr.iloc[0]))} "
          f"vs time-weighted {_pct(g['annualized'])}.", "",
          "### Estimated", "",
          f"- Model-net return assumes {cfg.fee.describe()}. Not a historical fact.",
          "- Allocation effect (Phase 3) uses a policy benchmark whose US/international split is inferred from "
          "returns; the construction error is shown as its own line.",
          "- Rolling-window alpha is a noisy estimate; the noise floor is stated in Phase 4.",
          ] + ([] if monthly else ["- Multi-factor (Carhart, FF5) loadings on ~30 annual observations are weakly identified."]) + ["",
          "### Unknown", "",
          f"- {len(flagged_ids)} statement(s) failed reconciliation and are excluded until re-read: "
          + (", ".join(flagged_ids) if flagged_ids else "none") + ".",
          f"- {len(unverified_ids)} unverified statement(s): " + (", ".join(unverified_ids[:12]) + (f", … and {len(unverified_ids) - 12} more" if len(unverified_ids) > 12 else "") if unverified_ids else "none") + ".",
          f"- {n_gaps} gap(s) in statement coverage (see coverage.md). No return is computed across a gap.",
          "- Selection vs trading split (Phase 3 Level 3) not run: requires security price history.",
          ] + ([] if monthly else ["- Intra-year drawdowns, monthly volatility, and 3-/5-year rolling alpha: not observable from annual statements. "
          "Available once Fidelity monthly PDFs are ingested."]) + [
          "- Anything before each account's first statement, and any account whose statements were never in the binder.", ""]

    # 3. reconciliation to claimed
    rec = reconciliation_rows(res2, comp, acc, g, n, pri, surv)
    pd.DataFrame(rec).to_csv(out / "reconciliation.csv", index=False)
    L += ["## 3. Reconciling to the claimed number", "",
          "The most common ways a self-calculated figure lands above the verified time-weighted composite, "
          "each computed from this data so the reader can see which (if any) explains the gap:", "",
          "| candidate calculation | value | why it differs |", "|---|---|---|"]
    for r in rec:
        v = f"**{_pct(r['value'])}**" if r["is_verified"] else _pct(r["value"])
        L.append(f"| {r['calculation']} | {v} | {r['why']} |")
    L.append("")

    # 4. stability
    sp = res4.subperiods.set_index(["model", "segment"])
    sm_ = hm if hm in set(sp.index.get_level_values(0)) else "CAPM"
    pre = sp.loc[(sm_, f"before {res4.config.split_year}")]; post = sp.loc[(sm_, f"{res4.config.split_year} onward")]
    diff = sp.loc[(sm_, "difference (post − pre)")]
    L += ["## 4. Is it stable?", "",
          f"{sm_} alpha before {res4.config.split_year}: {_pct(pre.alpha_annual)} (SE {_pct(pre.se_annual, 1)}, p = {pre.p:.2f}); from "
          f"{res4.config.split_year}: {_pct(post.alpha_annual)} (SE {_pct(post.se_annual, 1)}, p = {post.p:.2f}); "
          f"difference {_pct(diff.alpha_annual)} (p = {diff.p:.2f}).",
          "With standard errors that size, halves of a record with *constant* true alpha will differ by "
          f"more than {_pct(2 * diff.se_annual, 1)} about one time in twenty. "
          "Rolling detail and the noise floor for reading it are in phase4/summary.md.", ""]

    # 5. caveats
    L += ["## 5. Caveats that apply to every number above", "",
          ("- Monthly valuation. Not GIPS-compliant; GIPS-methodology-inspired." if monthly else
           "- Annual valuation only. Not GIPS-compliant; GIPS-methodology-inspired. Monthly data from the custodian "
           "would upgrade the recent decade."),
          f"- Benchmark ({bench_id}) and model fee were fixed before real data was examined (COMPOSITE_RULES.md §6, §9).",
          "- Nothing was interpolated, smoothed, or corrected. Flagged statements are excluded, not fixed.",
          "- In-sample. A 30-year record is one draw; the confidence interval, not the point estimate, is the claim.", ""]
    path = out / "REPORT.md"
    path.write_text("\n".join(L))
    return path
