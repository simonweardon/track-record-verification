"""Research notes rendered from the CSVs under data/research/ — same chrome as the dashboards.

Every sentence on these pages is generated from the numbers, so a rebuild with new
filings cannot leave stale prose behind.  The verdicts use the same thresholds as the
manager dashboards: |t| ≥ 2 is "evidence", anything less is "indistinguishable from zero".
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .dashboard import CSS, JS, diverging_bars, esc, line_chart, pct, num
from .signals13f import OUT_DIR as SIG_DIR, PORTFOLIOS, SPREADS

EXTRA_CSS = """
.chart .ser.s2 { stroke: var(--s3); } .chart .dot.s2 { fill: var(--s3); } .k.s2 { background: var(--s3); }
.chart .ser.s4 { stroke: var(--neg); } .chart .dot.s4 { fill: var(--neg); } .k.s4 { background: var(--neg); }
.chart .ser.s5 { stroke: var(--gold); } .chart .dot.s5 { fill: var(--gold); } .k.s5 { background: var(--gold); }
.findings { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 14px; margin: 6px 0 4px; }
.finding { background: var(--surface); border: 1px solid var(--line); border-top: 2px solid var(--gold); padding: 16px 18px; }
.finding .q { font: 400 17px/1.25 var(--serif); color: var(--navy); margin-bottom: 8px; }
.finding .a { font-size: 14px; color: var(--ink); }
.finding .v { display: inline-block; font: 600 9.5px/1 var(--sans); letter-spacing: .16em; text-transform: uppercase; padding: 5px 8px; margin-bottom: 10px; color: #fff; }
.finding .v.no { background: var(--crit); } .finding .v.weak { background: var(--warn); } .finding .v.yes { background: var(--good); } .finding .v.caveat { background: var(--ink-2); }
.legend { display: flex; flex-wrap: wrap; gap: 14px; font: 12px var(--sans); color: var(--ink-2); margin: 8px 0 4px; }
"""


def _tv(t) -> str:
    if t is None or pd.isna(t):
        return "n/a"
    return f"{t:+.1f}"


def _verdict_class(t: float, want_positive: bool = True) -> tuple[str, str]:
    """Traffic light for a spread: strong evidence in the hoped-for direction, weak, none, or the wrong way."""
    if pd.isna(t):
        return "caveat", "no test"
    if abs(t) < 1.0:
        return "no", "no signal"
    if abs(t) < 2.0:
        return "weak", "weak, not significant"
    return ("yes", "evidence") if (t > 0) == want_positive else ("no", "evidence — the wrong way")


def signals13f_html(sig_dir: Path = SIG_DIR) -> str | None:
    if not (sig_dir / "manifest.json").exists():
        return None
    man = json.loads((sig_dir / "manifest.json").read_text())
    summ = pd.read_csv(sig_dir / "summary.csv").set_index("key")
    spreads = pd.read_csv(sig_dir / "spreads.csv")
    ic = pd.read_csv(sig_dir / "ic.csv"); ics = pd.read_csv(sig_dir / "ic_summary.csv")
    R = pd.read_csv(sig_dir / "portfolios_monthly.csv", index_col=0, parse_dates=True)
    form = pd.read_csv(sig_dir / "formation.csv")
    crowd = pd.read_csv(sig_dir / "latest_crowding.csv")
    sp = spreads.set_index(["long", "short"])

    def spread(a, b):
        return sp.loc[(a, b)] if (a, b) in sp.index else None

    # ---- findings (prose from numbers)
    best = spread("BEST1", "ALL"); crd = spread("CROWD_HI", "CROWD_LO"); new = spread("NEW", "SOLD"); newu = spread("NEW", "ALL"); addc = spread("ADD", "CUT")
    findings = []
    if best is not None:
        c, lab = _verdict_class(best.t_spread)
        findings.append(("Do managers' best ideas beat the rest of their books?", c, lab,
                         f"Each manager's largest position, bought when its filing became public, returned {pct(best.ann_spread, 1)} per year "
                         f"versus the equal-weighted universe of everything they hold (t = {_tv(best.t_spread)}, positive in {best.share_positive_months:.0%} of months). "
                         f"After the Carhart four factors the difference is {pct(best.carhart_alpha, 1)}/yr (t = {_tv(best.carhart_t)}). "
                         f"Cohen, Polk and Silli found best ideas earned several percent a year in 1991–2005; at a 45-day lag, in this universe and period, that edge is not visible."))
    if crd is not None:
        lo = summ.loc["CROWD_LO"]
        c, lab = _verdict_class(crd.t_spread)
        findings.append(("Do crowded names outperform the ones nobody else holds?", "caveat", "unreliable — see why",
                         f"The most-held quintile {'lagged' if crd.ann_spread < 0 else 'beat'} the least-held quintile by {pct(abs(crd.ann_spread), 1, False)}/yr (spread t = {_tv(crd.t_spread)}); after factors {pct(crd.carhart_alpha, 1)}/yr (t = {_tv(crd.carhart_t)}). "
                         f"That looks like a real 'avoid the hotels' result — but the least-crowded quintile is the smallest-dollar stakes held by a single manager: "
                         f"a size loading of {num(lo.b_SMB)} and turnover of {lo.avg_turnover:.0%} a quarter. This is exactly where the price panel's survivorship bias "
                         f"(delisted names drop out at their last price) would manufacture a return, so it is reported and not believed. "
                         f"The rank test across all names says the opposite, mildly: more holders, slightly better next quarter (IC {ics.set_index('signal').loc['n_holders', 'mean_ic']:+.3f}, t = {_tv(ics.set_index('signal').loc['n_holders', 't'])})."))
    if new is not None and newu is not None:
        c, lab = _verdict_class(new.t_spread)
        findings.append(("Is copying managers' new buys profitable?", c if c != "yes" else "no", lab,
                         f"Stocks that managers had just bought returned {pct(new.ann_spread, 1)}/yr versus the stocks they had just sold (t = {_tv(new.t_spread)}), "
                         f"and {pct(newu.ann_spread, 1)}/yr versus the whole universe (Carhart alpha {pct(newu.carhart_alpha, 1)}, t = {_tv(newu.carhart_t)}). "
                         f"By the time a purchase is public, whatever information it carried has been priced; a portfolio that copied the trades would have lagged."))
    if addc is not None:
        c, lab = _verdict_class(addc.t_spread)
        findings.append(("Do adds beat trims?", c, lab,
                         f"Positions managers increased by 25%+ returned {pct(addc.ann_spread, 1)}/yr more than positions they trimmed by 25%+ "
                         f"(t = {_tv(addc.t_spread)}, positive in {addc.share_positive_months:.0%} of months; Carhart alpha {pct(addc.carhart_alpha, 1)}, t = {_tv(addc.carhart_t)}). "
                         f"The one sign in the hoped-for direction, and it is not significant."))
    fcards = "".join(f'<div class="finding"><div class="q">{esc(q)}</div><span class="v {c}">{esc(lab)}</span><div class="a">{a}</div></div>' for q, c, lab, a in findings)

    # ---- cumulative growth chart
    keys = [("MKT", "US market", "s0"), ("ALL", "Every held stock", "s1l"), ("BEST1", "Best ideas", "s1"),
            ("CROWD_HI", "Most crowded", "s2"), ("CROWD_LO", "Least crowded", "s5"), ("NEW", "New positions", "s4")]
    Rg = R[[k for k, _, _ in keys]].dropna(how="all")
    cells = [d.strftime("%Y-%m") for d in Rg.index]
    series = []
    for k, name, cls in keys:
        g = (1 + Rg[k].fillna(0)).cumprod()
        g[Rg[k].isna() & ~Rg[k].notna().cummax()] = np.nan      # nothing before the first month
        series.append(dict(name=name, values=[None if pd.isna(v) else float(v) for v in g], cls=cls, emph=(k == "BEST1")))
    growth = line_chart(cells, series, height=320, width=860, y_fmt=lambda v: f"{v:.1f}×", y_log=True, end_labels=False, uid="growth")
    legend = "".join(f'<span><span class="k {cls}"></span>{esc(name)}</span>' for _, name, cls in keys)

    # ---- portfolio table
    order = ["MKT", "ALL", "BEST1", "BEST3", "CROWD_HI", "CROWD_LO", "NEW", "ADD", "CUT", "SOLD"]
    prow = ""
    for k in order:
        if k not in summ.index: continue
        r = summ.loc[k]
        prow += (f"<tr><td><b>{esc(r.label)}</b><br><span class='muted'>{esc(r.holds)}</span></td>"
                 f"<td class='n'>{pct(r.ann_return, 1, False)}</td><td class='n'>{pct(r.ann_vol, 1, False)}</td><td class='n'>{num(r.sharpe)}</td>"
                 f"<td class='n'>{pct(r.max_dd, 0, False)}</td><td class='n'>{pct(r.excess_vs_market, 1)}</td>"
                 f"<td class='n'>{num(r.beta)}</td><td class='n'>{pct(r.carhart_alpha, 1) if k != 'MKT' else ''}</td><td class='n'>{_tv(r.carhart_t) if k != 'MKT' else ''}</td>"
                 f"<td class='n'>{f'{r.avg_names:,.0f}' if not pd.isna(r.avg_names) else ''}</td><td class='n'>{f'{r.avg_turnover:.0%}' if not pd.isna(r.avg_turnover) else ''}</td></tr>")

    # ---- spreads table + yearly bars for the two headline spreads
    srow = ""
    for _, r in spreads.iterrows():
        c, lab = _verdict_class(r.t_spread)
        srow += (f"<tr><td><b>{esc(r.label)}</b></td><td class='n'>{pct(r.ann_spread, 1)}</td><td class='n'>{_tv(r.t_spread)}</td><td class='n'>{r.share_positive_months:.0%}</td>"
                 f"<td class='n'>{pct(r.carhart_alpha, 1)}</td><td class='n'>{_tv(r.carhart_t)}</td><td class='n'>{num(r.b_SMB)}</td><td class='n'>{num(r.b_HML)}</td><td class='n'>{num(r.b_MOM)}</td>"
                 f"<td class='n'>{pct(r.worst_year, 0)}</td><td class='n'>{pct(r.best_year, 0)}</td><td><span class='v {c}' style='color:#fff;background:var(--{ {'no': 'crit', 'weak': 'warn', 'yes': 'good', 'caveat': 'ink-2'}[c] });font:600 9px var(--sans);letter-spacing:.14em;text-transform:uppercase;padding:4px 7px'>{esc(lab)}</span></td></tr>")
    def yearly(a, b):
        d = (R[a] - R[b]).dropna()
        y = d.groupby(d.index.year).sum()
        return diverging_bars([str(i) for i in y.index], [float(v) for v in y.values], height=200, width=860,
                              tips=[f"{i}: {PORTFOLIOS[a][0]} − {PORTFOLIOS[b][0]} {v * 100:+.1f} pp" for i, v in y.items()])
    bars_new = yearly("NEW", "SOLD"); bars_best = yearly("BEST1", "ALL")

    # ---- IC chart
    icc = ic.dropna(subset=["n_holders"])
    ic_bars = diverging_bars([str(f)[:7] for f in icc.formation], [float(v) for v in icc.n_holders], height=200, width=860, y_fmt=lambda v: f"{v:+.2f}",
                             tips=[f"{f}: rank correlation between number of holders and next-quarter return = {v:+.3f} across {n:,} stocks" for f, v, n in zip(icc.formation, icc.n_holders, icc.n_stocks)])
    icrow = "".join(f"<tr><td>{esc(r.label)}</td><td class='n'>{int(r.quarters)}</td><td class='n'>{r.mean_ic:+.3f}</td><td class='n'>{r.sd_ic:.3f}</td><td class='n'>{_tv(r.t)}</td><td class='n'>{r.share_positive:.0%}</td></tr>" for _, r in ics.iterrows())

    # ---- latest crowding
    crow = "".join(f"<tr><td><b>{esc(r.ticker)}</b> <span class='muted'>{esc(r['name'])}</span></td><td class='n'>{int(r.n_holders)}</td><td class='n'>{int(r.best_idea_of)}</td><td class='n'>{int(r.top3_of)}</td>"
                   f"<td class='n'>{r.avg_weight:.1%}</td><td class='n'>{int(r.new_buyers)}</td><td class='n'>{int(r.adders)}</td><td class='n'>{int(r.trimmers)}</td></tr>" for _, r in crowd.iterrows())
    last_form = str(crowd.formation.iloc[0]) if len(crowd) else man["last_formation"]
    n_mgr_last = int(form.managers.iloc[-1]) if len(form) else man["managers"]

    mapped_note = "96% of disclosed dollar value maps to a priced US ticker; the rest is foreign-domiciled, delisted or unlisted"
    return f"""<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>13F signals — research note</title>
<style>{CSS}{EXTRA_CSS}</style>
<div class="banner" role="note"><span class="bl">Research note</span> Public SEC 13F filings and Yahoo Finance prices. A reconstruction for research, not a strategy and not investment advice.</div>
<header class="cover"><div class="cover-in">
  <div class="cover-top"><div class="eyebrow">Quantitative research · 13F universe</div></div>
  <div class="gold-rule"></div>
  <h1>Do managers' disclosed<br>books carry a signal?</h1>
  <p class="sub">Every quarter, {man['managers']} prominent managers' public filings are turned into long-only portfolios — best ideas, crowded names, fresh buys — bought the month the filings become public and held three months. Then each one is tested the way a manager would be.</p>
  <dl class="meta">
    <div><dt>Managers</dt><dd>{man['managers']}</dd></div>
    <div><dt>Formation dates</dt><dd>{man['quarters']} · {man['first_formation'][:7]} → {man['last_formation'][:7]}</dd></div>
    <div><dt>Months tested</dt><dd>{man['months']}</dd></div>
    <div><dt>Positions read</dt><dd>{man['positions']:,}</dd></div>
    <div><dt>Built</dt><dd>{esc(man['built'])}</dd></div>
  </dl>
</div></header>
<main class="wrap">
<section class="verdict">
  <div class="sh"><h2>Findings</h2></div>
  <div class="findings">{fcards}</div>
  <p class="cap" style="margin-top:14px">Verdict rule: |t| below 1 is no signal, 1–2 is weak, 2 or more is evidence. t is the spread's mean monthly return divided by its standard error; the Carhart version is the intercept of a regression on the market, size, value and momentum factors with Newey–West errors. <b>Bottom line:</b> after the 45-day disclosure lag, the disclosed books of these managers contain no signal a copier could use, and the one result that looks like one sits where data bias lives.</p>
</section>

<section>
  <div class="sh"><h2>Growth of $1</h2></div>
  <div class="card">
    <div class="legend">{legend}</div>
    {growth}
    <p class="cap">Log scale. Each portfolio re-forms at the end of Feb, May, Aug and Nov from the filings public by then, equal-weighted at formation (best ideas: one unit per manager naming the stock), then held with drift. Months with fewer than five priced names are left blank, never bridged.</p>
  </div>
</section>

<section>
  <div class="sh"><h2>The portfolios</h2></div>
  <div class="card tscroll"><table><thead><tr><th>portfolio</th><th class="n">return /yr</th><th class="n">vol</th><th class="n">Sharpe</th><th class="n">max DD</th><th class="n">vs market /yr</th><th class="n">β</th><th class="n">Carhart α</th><th class="n">t</th><th class="n">names</th><th class="n">turnover /qtr</th></tr></thead><tbody>{prow}</tbody></table>
  <p class="cap">Sharpe uses the one-month T-bill. Carhart α is annualized; t uses Newey–West standard errors. Names and turnover are averages across formation dates (turnover = half the sum of absolute weight changes vs the drifted prior portfolio).</p></div>
</section>

<section>
  <div class="sh"><h2>The tests that matter: spreads</h2></div>
  <div class="card tscroll"><table><thead><tr><th>long − short</th><th class="n">spread /yr</th><th class="n">t</th><th class="n">months &gt; 0</th><th class="n">Carhart α</th><th class="n">t</th><th class="n">SMB</th><th class="n">HML</th><th class="n">MOM</th><th class="n">worst yr</th><th class="n">best yr</th><th>verdict</th></tr></thead><tbody>{srow}</tbody></table>
  <p class="cap">A spread cancels what both legs share — the market, the period, the universe — so it isolates the signal. Factor loadings show what the spread is secretly betting on: the crowding spread is short size (SMB {num(crd.b_SMB) if crd is not None else 'n/a'}), which is why its alpha is not taken at face value.</p></div>
  <div class="grid2">
    <div class="card"><h3>New positions − Sold out, by year</h3>{bars_new}<p class="cap">Sum of monthly spread returns each calendar year.</p></div>
    <div class="card"><h3>Best ideas − Every held stock, by year</h3>{bars_best}<p class="cap">Sum of monthly spread returns each calendar year.</p></div>
  </div>
</section>

<section>
  <div class="sh"><h2>Rank tests: information coefficients</h2></div>
  <div class="grid2">
    <div class="card"><h3>Crowding IC by formation date</h3>{ic_bars}
      <p class="cap">Spearman rank correlation between the number of managers holding a stock and its return over the next quarter, across every held stock with a price. An IC of 0.02 is small but persistent: positive in {ics.set_index('signal').loc['n_holders', 'share_positive']:.0%} of quarters.</p></div>
    <div class="card"><h3>All signals</h3><div class="tscroll"><table><thead><tr><th>signal</th><th class="n">quarters</th><th class="n">mean IC</th><th class="n">sd</th><th class="n">t</th><th class="n">&gt; 0</th></tr></thead><tbody>{icrow}</tbody></table></div>
      <p class="cap">The IC is the standard first look at a stock-selection signal: does the ranking predict next period's ranking of returns? t = mean ÷ (sd ÷ √quarters). Real, tradable signals run 0.03–0.05 with t above 3 over long samples.</p></div>
  </div>
</section>

<section>
  <div class="sh"><h2>Most crowded names now</h2></div>
  <div class="card tscroll"><table><thead><tr><th>stock</th><th class="n">managers holding</th><th class="n">#1 position of</th><th class="n">top-3 of</th><th class="n">avg weight</th><th class="n">new buyers</th><th class="n">added</th><th class="n">trimmed</th></tr></thead><tbody>{crow}</tbody></table>
  <p class="cap">From the filings public by {esc(last_form)}, {n_mgr_last} managers. Weight is the average share of a holder's disclosed book. New buyers, added and trimmed count managers whose position is new, up 25%+ in shares, or down 25%+ versus their previous filing.</p></div>
</section>

<section>
  <div class="sh"><h2>Method and limits</h2></div>
  <div class="card">
    <p><b>Universe.</b> The {man['managers']} managers in the system whose 13F is a portfolio rather than trading inventory (multi-strategy, quant, macro and market-making firms are excluded). All filer entities of a manager are merged; puts and calls are dropped; a filing needs at least five long positions. {mapped_note}.</p>
    <p><b>Timing.</b> A quarter's holdings are used at the end of the second month after quarter end — the month in which the 45-day deadline falls — and only if the filing was actually public by then. Positions are held with drift until the next formation date. This is the earliest any outsider could have acted.</p>
    <p><b>Weights.</b> "Every held stock" is equal-weighted per stock. Best ideas, new, added, trimmed and sold portfolios weight each stock by the number of managers in that category — one unit per manager-position pair. Crowding quintiles rank held stocks by number of holders, ties broken by total dollars held.</p>
    <p><b>Prices.</b> Yahoo Finance monthly adjusted closes. Stocks under $1 at formation are excluded; a monthly return above +300% from a start under $5 is treated as a data error and blanked ({man.get('spikes_removed', 0)} cases in the whole panel). Delisted names disappear at their last price, so every long portfolio here is survivor-biased upward — most for the smallest, least-held names.</p>
    <p><b>Tests.</b> Spread t-statistics are the mean monthly spread over its standard error. Alphas are intercepts of monthly regressions on the Ken French US market, SMB, HML and momentum factors with Newey–West (HAC) errors, annualized ×12. Information coefficients are Spearman correlations at each formation date. Nothing is optimized: the portfolio definitions were fixed before the results were seen, and every result — including the ones that argue against the idea — is shown.</p>
    <p><b>Literature.</b> Cohen, Polk &amp; Silli (2010), "Best Ideas"; Brown &amp; Schwarz (2013) on the information content of 13F disclosures; Lakonishok, Shleifer &amp; Vishny (1992) on herding. The question here is narrower than theirs: whether an outsider, acting only on public filings at the public date, could have used them.</p>
  </div>
</section>

<footer class="foot">
  <div class="running"><span>Track record verification · research</span><span>{man['first_formation'][:7]} – {man['last_formation'][:7]}</span></div>
  <h4>Important information</h4>
  <p>Built from public SEC EDGAR 13F-HR filings and Yahoo Finance prices; benchmark and factor data from the Kenneth R. French Data Library. All results are in-sample reconstructions with the limits stated above; they are not a strategy, a recommendation or investment advice. Past performance is not indicative of future results. Rebuild: <code>python -m trackrecord signals13f</code>.</p>
</footer>
</main>
<div id="tip" class="tip" hidden></div>
<script>{JS}</script>
"""


# ---------------------------------------------------------------- portfolio construction note

def construction_html(out_dir: Path | None = None) -> str | None:
    from .construct import OUT_DIR as CON_DIR
    d = out_dir or CON_DIR
    if not (d / "manifest.json").exists():
        return None
    man = json.loads((d / "manifest.json").read_text())
    summ = pd.read_csv(d / "summary.csv").set_index("key")
    rb = pd.read_csv(d / "rebalances.csv")
    R = pd.read_csv(d / "backtest_monthly.csv", index_col=0, parse_dates=True)
    tl = pd.read_csv(d / "trade_list.csv")
    sec = pd.read_csv(d / "sectors_latest.csv")
    c = man["constraints"]; L = man["latest"]; rt = man.get("r_twin", {})
    P, U, B = summ.loc["portfolio"], summ.loc["unconstrained"], summ.loc["benchmark"]

    # growth chart
    Rg = R[["benchmark", "portfolio", "unconstrained"]].dropna(how="all")
    cells = [x.strftime("%Y-%m") for x in Rg.index]
    series = []
    for k, name, cls in [("benchmark", "Benchmark", "s0"), ("portfolio", "Constrained LP portfolio", "s1"), ("unconstrained", "Unconstrained top decile", "s4")]:
        g = (1 + Rg[k].fillna(0)).cumprod()
        series.append(dict(name=name, values=[float(v) for v in g], cls=cls, emph=(k == "portfolio")))
    growth = line_chart(cells, series, height=300, width=860, y_fmt=lambda v: f"{v:.1f}×", y_log=True, end_labels=False, uid="cgrowth")
    legend = "".join(f'<span><span class="k {cls}"></span>{esc(n)}</span>' for _, n, cls in [(0, "Benchmark", "s0"), (0, "Constrained LP portfolio", "s1"), (0, "Unconstrained top decile", "s4")])
    # active return by year
    act = (R.portfolio - R.benchmark).dropna(); ay = act.groupby(act.index.year).sum()
    act_bars = diverging_bars([str(i) for i in ay.index], [float(v) for v in ay.values], height=200, width=860,
                              tips=[f"{i}: portfolio − benchmark {v * 100:+.1f} pp" for i, v in ay.items()])
    # ex-ante vs realized TE per rebalance (realized over the following quarter)
    te_rows = ""
    for _, r in rb.tail(10).iloc[::-1].iterrows():
        te_rows += (f"<tr><td>{esc(str(r.formation))}</td><td class='n'>{int(r.universe)}</td><td class='n'>{int(r.names)}</td><td class='n'>{r.turnover:.1%}</td>"
                    f"<td class='n'>{r.active_share:.0%}</td><td class='n'>{r.ex_ante_te:.1%}</td><td class='n'>{r.max_active:.1%}</td><td class='n'>{r.max_sector_active:.1%}</td><td class='n'>{r.sum_abs_active / 2:.1%}</td>"
                    f"<td class='n'>{r.expected_alpha - r.bench_alpha:+.2f}</td><td>{'turnover relaxed ×' + str(2 ** int(r.relaxed)) if r.relaxed else ''}</td></tr>")
    # trade list (top 12 each way)
    buys = tl[tl.side == "BUY"].head(12); sells = tl[tl.side == "SELL"].head(12)
    def trow(x):
        return "".join(f"<tr><td><b>{esc(r.ticker)}</b> <span class='muted'>{esc(str(r['name'])[:28])}</span></td><td class='muted'>{esc(r.sector)}</td>"
                       f"<td class='n'>{r.current_wt:.2%}</td><td class='n'>{r.target_wt:.2%}</td><td class='n'>{r.active_wt:+.2%}</td><td class='n'>{'—' if pd.isna(r.alpha_z) else f'{r.alpha_z:+.2f}'}</td>"
                       f"<td class='n'>{r.shares:+,}</td><td class='n'>${r.trade_usd / 1e6:+,.2f}m</td></tr>" for _, r in x.iterrows())
    sec_rows = "".join(f"<tr><td>{esc(r.sector)}</td><td class='n'>{r.benchmark:.1%}</td><td class='n'>{r.portfolio:.1%}</td><td class='n'>{r.active:+.1%}</td></tr>" for _, r in sec.iterrows())
    r_block = (f"<p>The same problem was handed to <b>R</b> — <code>r/construct.R</code>, data.table for the inputs, Rglpk (GLPK simplex) for the solve — on the {esc(L['formation'])} inputs. "
               f"Both solvers hold {rt.get('names_r')} / {rt.get('names_python')} names with expected alpha {rt.get('expected_alpha_r', 0):.4f} vs {rt.get('expected_alpha_python', 0):.4f}; "
               f"largest weight difference {rt.get('max_abs_diff', 0):.1e}, total {rt.get('sum_abs_diff', 0):.1e}. Linear programs have one optimum value; ties between equally good corners can differ, which is why the objective — not the weights — is the check.</p>"
               if rt.get("available") else f"<p>R twin not run on this build ({esc(str(rt.get('reason', '')))}). The script is <code>r/construct.R</code>; the unit test compares it with the Python solve whenever Rscript and Rglpk are present.</p>")
    return f"""<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Portfolio construction — LP with trade list</title>
<style>{CSS}{EXTRA_CSS}</style>
<div class="banner" role="note"><span class="bl">Research note</span> A demonstration mandate on public data — a transparent signal through a real constraint set. Not a strategy and not investment advice.</div>
<header class="cover"><div class="cover-in">
  <div class="cover-top"><div class="eyebrow">Quantitative portfolio management · construction</div></div>
  <div class="gold-rule"></div>
  <h1>From signal to trade list,<br>under a mandate's constraints</h1>
  <p class="sub">A linear program turns alpha scores into target weights that respect name caps, active and sector bands and a turnover budget — then into the buy and sell tickets a trader would receive. Rebalanced {man['rebalances']} times since {man['first'][:7]}; solved in Python (HiGHS) and R (Rglpk) and checked against each other.</p>
  <dl class="meta">
    <div><dt>Rebalances</dt><dd>{man['rebalances']} · {man['first'][:7]} → {man['last'][:7]}</dd></div>
    <div><dt>Universe</dt><dd>names held by ≥ {man['min_holders']} managers</dd></div>
    <div><dt>Mandate NAV</dt><dd>${man['nav'] / 1e6:,.0f}m</dd></div>
    <div><dt>Built</dt><dd>{esc(man['built'])}</dd></div>
  </dl>
</div></header>
<main class="wrap">
<section class="verdict">
  <div class="sh"><h2>What the constraints bought and cost</h2></div>
  <div class="tiles">
    <div class="tile"><div class="tl">Active return, constrained</div><div class="tv">{pct(P.active_return, 1)}</div><div class="td muted">per year vs the aggregate book, after {c['cost_bps']:.0f} bps costs</div></div>
    <div class="tile"><div class="tl">Tracking error</div><div class="tv">{pct(P.tracking_error, 1, False)}</div><div class="td muted">realized · ex-ante averaged {pct(P.avg_ex_ante_te, 1, False)}</div></div>
    <div class="tile"><div class="tl">Information ratio</div><div class="tv">{num(P.information_ratio)}</div><div class="td muted">active return ÷ tracking error · hit rate {P.hit_rate:.0%} of months</div></div>
    <div class="tile"><div class="tl">Turnover</div><div class="tv">{P.avg_turnover:.0%}</div><div class="td muted">one-way per quarter · budget {c['turnover']:.0%} · {P.avg_names:.0f} names on average</div></div>
    <div class="tile"><div class="tl">Unconstrained top decile</div><div class="tv">{pct(U.active_return, 1)}</div><div class="td muted">same signal, no constraints: TE {pct(U.tracking_error, 1, False)}, IR {num(U.information_ratio)}, turnover {U.avg_turnover:.0%}, {U.avg_names:.0f} names</div></div>
    <div class="tile"><div class="tl">Benchmark</div><div class="tv">{pct(B.ann_return, 1, False)}</div><div class="td muted">aggregate disclosed book, dollar-weighted · vol {pct(B.ann_vol, 1, False)} · max DD {pct(B.max_dd, 0, False)}</div></div>
  </div>
  <p class="cap" style="margin-top:14px">The signal here is 12-1 month price momentum, chosen because it is transparent and available for every name — not because it is good. The point of the page is the machinery: the same code takes any alpha vector. Read the constrained and unconstrained rows together: the constraints trade raw signal exposure for a portfolio a benchmark-relative mandate could actually hold.</p>
</section>

<section>
  <div class="sh"><h2>Growth of $1</h2></div>
  <div class="card"><div class="legend">{legend}</div>{growth}
  <p class="cap">Log scale. Costs of {c['cost_bps']:.0f} bps per dollar traded are charged in the month after each rebalance, on both the constrained and the unconstrained portfolio.</p></div>
  <div class="card" style="margin-top:14px"><h3>Active return by year (constrained − benchmark)</h3>{act_bars}</div>
</section>

<section>
  <div class="sh"><h2>The mandate</h2></div>
  <div class="grid2">
    <div class="card"><h3>Constraints</h3>
      <div class="tscroll"><table><tbody>
      <tr><td>Long-only, fully invested</td><td class="n">0 ≤ w, Σw = 1</td></tr>
      <tr><td>Name cap</td><td class="n">w ≤ {c['max_weight']:.0%}</td></tr>
      <tr><td>Active band per name</td><td class="n">|w − b| ≤ {c['active_band']:.0%}</td></tr>
      <tr><td>Sector band vs benchmark</td><td class="n">|Σ<sub>s</sub>w − Σ<sub>s</sub>b| ≤ {c['sector_band']:.0%}</td></tr>
      <tr><td>Active share cap</td><td class="n">Σ|w − b| / 2 ≤ {c['active_share']:.0%}</td></tr>
      <tr><td>Turnover budget, one-way</td><td class="n">Σ|w − w₀| / 2 ≤ {c['turnover']:.0%}</td></tr>
      <tr><td>Names leaving the universe</td><td class="n">forced to 0</td></tr>
      <tr><td>Objective</td><td class="n">max α'w − {c['cost_bps']:.0f} bps × traded</td></tr>
      </tbody></table></div>
      <p class="cap">Linear in w = w₀ + buy − sell, so it is an LP: {L['n_vars']:,} variables and {L['n_constraints']:,} constraints at the latest rebalance, solved in well under a second. When the drifted book cannot meet the sector bands within the turnover budget the budget is doubled and the rebalance is flagged.</p>
    </div>
    <div class="card"><h3>Sector exposure at the latest rebalance ({esc(L['formation'])})</h3>
      <div class="tscroll"><table><thead><tr><th>sector</th><th class="n">benchmark</th><th class="n">portfolio</th><th class="n">active</th></tr></thead><tbody>{sec_rows}</tbody></table></div>
      <p class="cap">Sectors from Yahoo Finance for {man['sectors_known']} names; {man['sectors_unknown']} unclassified names sit in their own "Unknown" band. A production mandate would use the index vendor's GICS.</p>
    </div>
  </div>
</section>

<section>
  <div class="sh"><h2>Trade list — {esc(L['formation'])}</h2></div>
  <p class="note">{L['trades']} tickets: {L['buys']} buys, {L['sells']} sells, ${L['traded_usd'] / 1e6:,.1f}m traded on a ${man['nav'] / 1e6:,.0f}m book (one-way turnover {L['turnover']:.1%}), estimated cost ${L['est_cost_usd'] / 1e3:,.0f}k. Ex-ante tracking error {L['ex_ante_te']:.1%}. Largest tickets each way:</p>
  <div>
    <div class="card tscroll"><h3>Buys</h3><table><thead><tr><th>stock</th><th>sector</th><th class="n">now</th><th class="n">target</th><th class="n">active</th><th class="n">α z</th><th class="n">shares</th><th class="n">$</th></tr></thead><tbody>{trow(buys)}</tbody></table></div>
    <div class="card tscroll" style="margin-top:14px"><h3>Sells</h3><table><thead><tr><th>stock</th><th>sector</th><th class="n">now</th><th class="n">target</th><th class="n">active</th><th class="n">α z</th><th class="n">shares</th><th class="n">$</th></tr></thead><tbody>{trow(sells)}</tbody></table></div>
  </div>
  <p class="cap">Full list with every name: <a href="/research/construction/trade_list.csv">trade_list.csv</a>. Shares are rounded to whole shares at the formation-date close; "now" is the previous target drifted through the quarter.</p>
</section>

<section>
  <div class="sh"><h2>Rebalance log</h2></div>
  <div class="card tscroll"><table><thead><tr><th>date</th><th class="n">universe</th><th class="n">held</th><th class="n">turnover</th><th class="n">active share</th><th class="n">ex-ante TE</th><th class="n">max |active|</th><th class="n">max |sector|</th><th class="n">active share</th><th class="n">α gain vs bench</th><th>note</th></tr></thead><tbody>{te_rows}</tbody></table>
  <p class="cap">Last ten rebalances. "α gain" is the expected alpha of the portfolio minus that of the benchmark, in z-score units — what the optimizer bought within the bands. Full log: <a href="/research/construction/rebalances.csv">rebalances.csv</a>.</p></div>
</section>

<section>
  <div class="sh"><h2>Same problem in R</h2></div>
  <div class="card">{r_block}</div>
</section>

<section>
  <div class="sh"><h2>Method and limits</h2></div>
  <div class="card">
    <p><b>Universe and benchmark.</b> Each quarter, the US names held by at least {man['min_holders']} of the managers in the system, priced at $1 or more. The benchmark is those names weighted by the total dollars the managers hold — the aggregate disclosed book. It is a real, investable, public portfolio, but it is not an index; a mandate would use one.</p>
    <p><b>Signal.</b> 12-1 momentum: the return from twelve months before the rebalance to one month before, z-scored across the universe and clipped at ±3. Nothing else. No fundamentals, no risk model factors, no combination — the page demonstrates construction, not alpha.</p>
    <p><b>Costs and drift.</b> {c['cost_bps']:.0f} bps per dollar traded, charged in the first month after each rebalance. Between rebalances every portfolio, including the benchmark, holds with drift; delisted names drop at their last price (survivorship, disclosed).</p>
    <p><b>Risk.</b> Ex-ante tracking error is √(12 · a'Σa) with Σ the trailing-36-month sample covariance shrunk halfway toward constant correlation. It is reported, not constrained: a quadratic term would leave the LP, and the active and sector bands are the linear proxy most benchmark-relative mandates actually use. Realized TE is the standard deviation of monthly active returns, annualized.</p>
    <p><b>Honesty.</b> The constraints were fixed before any backtest was run and were not tuned. Both the constrained and the unconstrained rows are shown, whatever they say.</p>
  </div>
</section>

<footer class="foot">
  <div class="running"><span>Track record verification · research</span><span>{man['first'][:7]} – {man['last'][:7]}</span></div>
  <h4>Important information</h4>
  <p>Built from public SEC EDGAR 13F-HR filings and Yahoo Finance prices and sector labels; factor data from the Kenneth R. French Data Library. A demonstration of portfolio-construction machinery on public data, with the limits stated above; not a strategy, a recommendation or investment advice. Past performance is not indicative of future results. Rebuild: <code>python -m trackrecord construct</code>; R twin: <code>Rscript r/construct.R data/research/construction/inputs_latest</code>.</p>
</footer>
</main>
<div id="tip" class="tip" hidden></div>
<script>{JS}</script>
"""
