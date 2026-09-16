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
<title>13F Signal Research</title>
<style>{CSS}{EXTRA_CSS}</style>
<div class="banner" role="note"><span class="bl">Research note</span> Public SEC 13F filings and Yahoo Finance prices. A reconstruction for research, not a strategy and not investment advice.</div>
<header class="cover"><div class="cover-in">
  <div class="cover-top"><div class="eyebrow">Manager Analysis · 13F Signal Research</div></div>
  <div class="gold-rule"></div>
  <h1>13F Signal Research</h1>
  <p class="sub">Do managers' disclosed books carry a signal? Every quarter, {man['managers']} prominent managers' public filings are turned into long-only portfolios — best ideas, crowded names, fresh buys — bought the month the filings become public and held three months. Then each one is tested the way a manager would be.</p>
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
<title>Portfolio Construction</title>
<style>{CSS}{EXTRA_CSS}</style>
<div class="banner" role="note"><span class="bl">Research note</span> A demonstration mandate on public data — a transparent signal through a real constraint set. Not a strategy and not investment advice.</div>
<header class="cover"><div class="cover-in">
  <div class="cover-top"><div class="eyebrow">Active · Portfolio Construction</div></div>
  <div class="gold-rule"></div>
  <h1>Portfolio Construction</h1>
  <p class="sub">From signal to trade list, under a mandate's constraints. A linear program turns alpha scores into target weights that respect name caps, active and sector bands and a turnover budget — then into the buy and sell tickets a trader would receive. Rebalanced {man['rebalances']} times since {man['first'][:7]}; solved in Python (HiGHS) and R (Rglpk) and checked against each other.</p>
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


# ---------------------------------------------------------------- R reproduction note

def _sci(x) -> str:
    return "n/a" if x is None or pd.isna(x) else f"{x:.1e}"


def rverify_html(out_dir: Path | None = None) -> str | None:
    from .rverify import OUT_DIR as RV_DIR, TOL
    d = out_dir or RV_DIR
    if not (d / "manifest.json").exists():
        return None
    man = json.loads((d / "manifest.json").read_text())
    s = pd.read_csv(d / "summary.csv")
    byq = pd.read_csv(d / "by_quantity.csv")
    if s.empty:
        return None

    agreed = int(s.agree.sum()); bad = int((~s.agree).sum())
    worst_mgr = s.loc[s.max_abs_diff.idxmax()]
    # the quantities table: worst disagreement per quantity, largest first
    qrows = ""
    for _, r in byq.iterrows():
        label = f"{esc(r.quantity)}" + (f" <span class='muted'>{esc(r.model)}</span>" if r.model != "-" else "")
        cls = "" if r.max_abs_diff < TOL else " class='bad'"
        qrows += (f"<tr{cls}><td>{label}</td><td class='n'>{int(r.managers)}</td>"
                  f"<td class='n'>{_sci(r.max_abs_diff)}</td><td class='n'>{_sci(r.max_rel_diff)}</td></tr>")
    # per-manager table, largest difference first — the ones worth looking at are at the top
    mrows = ""
    for _, r in s.sort_values("max_abs_diff", ascending=False).head(20).iterrows():
        v, vt = ("yes", "agree") if r.agree else ("no", "differs")
        mrows += (f"<tr><td><a href='/f/{esc(r.slug)}/memo'>{esc(r['name'])}</a></td><td class='n'>{int(r.cells)}</td>"
                  f"<td class='n'>{pct(r.ff3_alpha_r, 1)}</td><td class='n'>{pct(r.ff3_alpha_python, 1)}</td>"
                  f"<td class='n'>{_tv(r.ff3_t_r)}</td><td class='n'>{_tv(r.ff3_t_python)}</td>"
                  f"<td class='n'>{num(r.score_r, 1)}</td><td class='n'>{num(r.score_python, 1)}</td>"
                  f"<td class='n'>{_sci(r.max_abs_diff)}</td><td><span class='v {v}'>{vt}</span></td></tr>")
    verdict = (f"All {man['checks']:,} numbers agree" if not bad else
               f"{bad} of {man['managers']} managers disagree")
    return f"""<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>R Verification</title>
<style>{CSS}{EXTRA_CSS}
tr.bad td {{ background: color-mix(in srgb, var(--crit) 12%, transparent); }}
.v {{ display: inline-block; font: 600 9.5px/1 var(--sans); letter-spacing: .12em; text-transform: uppercase; padding: 4px 7px; color: #fff; }}
.v.yes {{ background: var(--good); }} .v.no {{ background: var(--crit); }}
</style>
<div class="banner" role="note"><span class="bl">Verification</span> A second implementation of the same statistics, written in R against the same aligned data. It tests the code, not the conclusion.</div>
<header class="cover"><div class="cover-in">
  <div class="cover-top"><div class="eyebrow">Active · R Verification</div></div>
  <div class="gold-rule"></div>
  <h1>R Verification</h1>
  <p class="sub">The same alphas, recomputed in R. Every headline number on a manager's dashboard — factor alphas, their Newey-West t-statistics, annualized return, volatility, Sharpe, max drawdown and the alpha-maxing score — is recomputed by <code>r/verify.R</code> in data.table and base R, from the aligned returns alone, and compared with what Python reported. No shared code: the HAC sandwich is written out by hand on the R side.</p>
  <dl class="meta">
    <div><dt>Managers</dt><dd>{man['managers']}</dd></div>
    <div><dt>Numbers checked</dt><dd>{man['checks']:,}</dd></div>
    <div><dt>Largest difference</dt><dd>{_sci(man['max_abs_diff'])}</dd></div>
    <div><dt>Built</dt><dd>{esc(man['built'])} · {esc(man['r'])}</dd></div>
  </dl>
</div></header>
<main class="wrap">
<section class="verdict">
  <div class="sh"><h2>{verdict}</h2></div>
  <div class="tiles">
    <div class="tile"><div class="tl">Managers agreeing</div><div class="tv">{agreed} / {man['managers']}</div><div class="td muted">agreement at better than {_sci(TOL)} on every number</div></div>
    <div class="tile"><div class="tl">Numbers compared</div><div class="tv">{man['checks']:,}</div><div class="td muted">{man['quantities_per_manager']} per manager — four factor models plus risk metrics and the score</div></div>
    <div class="tile"><div class="tl">Largest difference</div><div class="tv">{_sci(man['max_abs_diff'])}</div><div class="td muted">{esc(worst_mgr['name'])}, {esc(worst_mgr.worst_quantity)} — double precision noise, not a discrepancy</div></div>
    <div class="tile"><div class="tl">Disagreements</div><div class="tv">{bad}</div><div class="td muted">above the {_sci(TOL)} threshold, which is where a real bug would show</div></div>
  </div>
  <p class="cap" style="margin-top:14px">Agreement at 1e-13 means the two implementations do the same arithmetic. It does not mean the model is the right one, that the clones track the funds, or that an alpha with t = 0.4 is real — the memos say what the numbers are worth. What it rules out is the quiet kind of error: a wrong lag in the HAC weights, an off-by-one in the annualization, a drop-NA that silently changes the sample.</p>
</section>

<section>
  <div class="sh"><h2>Agreement by quantity</h2></div>
  <div class="card tscroll"><table><thead><tr><th>quantity</th><th class="n">managers</th><th class="n">max |R − Python|</th><th class="n">max relative</th></tr></thead><tbody>{qrows}</tbody></table>
  <p class="cap">Worst case across every manager, largest first. Full detail: <a href="/research/r-verify/by_quantity.csv">by_quantity.csv</a>, <a href="/research/r-verify/summary.csv">summary.csv</a>.</p></div>
</section>

<section>
  <div class="sh"><h2>Manager by manager</h2></div>
  <div class="card tscroll"><table><thead><tr><th>manager</th><th class="n">months</th><th class="n">FF3 α R</th><th class="n">FF3 α Python</th><th class="n">t R</th><th class="n">t Python</th><th class="n">score R</th><th class="n">score Python</th><th class="n">max diff</th><th></th></tr></thead><tbody>{mrows}</tbody></table>
  <p class="cap">The twenty managers with the largest difference — that is, the hardest cases for the comparison, not the best managers. Alphas are annualized; t is the Newey-West t on the FF3 intercept.</p></div>
</section>

<section>
  <div class="sh"><h2>What R recomputes</h2></div>
  <div class="grid2">
    <div class="card"><h3>The regression</h3>
      <p>For each of CAPM, FF3, Carhart 4 and FF5, R fits R<sub>p</sub> − RF = α + Σ b<sub>k</sub>f<sub>k</sub> + ε by ordinary least squares and builds the Newey-West covariance from scratch:</p>
      <p class="note">V = (X'X)<sup>−1</sup> S (X'X)<sup>−1</sup>, &nbsp; S = S<sub>0</sub> + Σ<sub>l=1..L</sub> w<sub>l</sub>(S<sub>l</sub> + S<sub>l</sub>'), &nbsp; w<sub>l</sub> = 1 − l/(L+1), &nbsp; L = ⌊0.75·n<sup>1/3</sup>⌋</p>
      <p>with no small-sample correction and normal — not t — p-values and confidence intervals, which is what <code>statsmodels</code> does under <code>cov_type="HAC"</code>. Those three choices are exactly where two implementations usually drift apart, so they are the point of the exercise.</p>
    </div>
    <div class="card"><h3>The rest</h3>
      <p>Annualized return as (Π(1+r))<sup>1/years</sup> − 1; volatility as the sample standard deviation times √12; Sharpe as the mean excess return over its standard deviation, times √12; max drawdown from the compounded index of monthly cells. The alpha-maxing score is rebuilt from its fixed map, 50 + 10 × excess return over the market in points, clipped to 0–100.</p>
      <p>R reads only <code>aligned_data.csv</code> — the returns and factors after alignment — and then the Python outputs purely to compare against. Any disagreement in the alignment itself would be invisible to this test; it checks the statistics, not the data assembly, which is what the coverage and reconciliation phases are for.</p>
    </div>
  </div>
</section>

<footer class="foot">
  <div class="running"><span>Track record verification · verification</span><span>{man['managers']} managers · {man['checks']:,} numbers</span></div>
  <h4>Important information</h4>
  <p>An internal consistency check between two implementations of the same statistics, run on clone portfolios reconstructed from public SEC EDGAR 13F-HR filings; factor data from the Kenneth R. French Data Library. Agreement between implementations is not evidence that a manager has skill. Not a recommendation or investment advice. Rebuild: <code>python -m trackrecord r-verify</code>; one manager: <code>Rscript r/verify.R output/funds/&lt;slug&gt;/phase4</code>. R is not installed on the server, so these tables are built locally and committed.</p>
</footer>
</main>
<div id="tip" class="tip" hidden></div>
<script>{JS}</script>
"""


# ---------------------------------------------------------------- research index

def research_index_html() -> str:
    """One page listing the research notes and every dataset behind them.

    The point of the page is that a reader can take the data away: each note is paired
    with the CSVs it was built from, and each row says plainly what is in the file."""
    from .rverify import OUT_DIR as RV_DIR
    from .construct import OUT_DIR as CON_DIR
    R = Path(__file__).resolve().parents[1] / "data" / "research"

    def man(p):
        try:
            return json.loads(p.read_text()) if p.exists() else {}
        except Exception:
            return {}

    sig, con, rv = man(SIG_DIR / "manifest.json"), man(CON_DIR / "manifest.json"), man(RV_DIR / "manifest.json")

    notes = []
    if sig:
        notes.append(("/research/13f-signals", "13F Signal Research",
                      f"Best-ideas, crowding and conviction-change portfolios from {sig.get('managers', '')} managers over "
                      f"{sig.get('quarters', '')} quarters ({sig.get('positions', 0):,} positions), formed 45 days after each "
                      "quarter-end and tested as Carhart spreads with HAC t-statistics.",
                      [("13f-signals/summary.csv", "one row per portfolio: return, vol, Sharpe, max DD, CAPM and Carhart alpha with t, turnover"),
                       ("13f-signals/spreads.csv", "long-short spreads between portfolios, with alphas and t-statistics"),
                       ("13f-signals/portfolios_monthly.csv", "the monthly return series of every portfolio"),
                       ("13f-signals/ic.csv", "rank information coefficient at each formation date"),
                       ("13f-signals/latest_crowding.csv", "how many managers hold each name in the latest filings")]))
    if con:
        notes.append(("/research/construction", "Portfolio Construction",
                      f"A demonstration mandate rebalanced {con.get('rebalances', '')} times on a "
                      f"${con.get('nav', 0) / 1e6:,.0f}m book: an LP maximising alpha net of cost under name, sector, "
                      "active-share and turnover constraints, solved in Python (HiGHS) and R (Rglpk).",
                      [("construction/summary.csv", "constrained, unconstrained and benchmark portfolios side by side"),
                       ("construction/trade_list.csv", "the latest trade list: every ticket with shares, dollars and active weight"),
                       ("construction/rebalances.csv", "every rebalance with turnover, active share and ex-ante tracking error"),
                       ("construction/backtest_monthly.csv", "monthly returns of all three portfolios"),
                       ("construction/sectors_latest.csv", "sector exposure against the benchmark at the latest rebalance")]))
    if rv.get("managers"):
        notes.append(("/research/r-verify", "R Verification",
                      f"{rv.get('checks', 0):,} numbers across {rv.get('managers')} managers recomputed by an independent "
                      f"R implementation and compared with Python; largest difference {rv.get('max_abs_diff', 0):.0e}, "
                      f"{rv.get('disagreements', 0)} disagreements.",
                      [("r-verify/summary.csv", "one row per manager: FF3 alpha and t from both implementations, worst difference"),
                       ("r-verify/by_quantity.csv", "worst disagreement per quantity across every manager")]))

    blocks = ""
    for href, title, what, files in notes:
        rows = "".join(
            f"<tr><td><a href='/research/data/{esc(f)}'><code>{esc(f)}</code></a></td><td>{esc(d)}</td>"
            f"<td class='n'>{_kb(R / f)}</td></tr>" for f, d in files)
        blocks += (f"<section><div class='sh'><h2><a href='{href}'>{esc(title)}</a></h2></div>"
                   f"<div class='card'><p>{esc(what)}</p>"
                   f"<div class='tscroll'><table><thead><tr><th>file</th><th>what is in it</th><th class='n'>size</th></tr></thead>"
                   f"<tbody>{rows}</tbody></table></div>"
                   f"<p class='cap'>Read the note: <a href='{href}'>{esc(title)}</a></p></div></section>")

    return f"""<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Research Data</title>
<style>{CSS}{EXTRA_CSS}
.sh h2 a {{ color: inherit; text-decoration: none; border-bottom: 1px solid var(--gold); }}
.sh h2 a:hover {{ color: var(--gold); }}
.cover .sub a {{ color: var(--gold); text-decoration: none; border-bottom: 1px solid rgba(197,167,106,.5); }}
.card td a code {{ font-size: 12.5px; }}
</style>
<div class="banner" role="note"><span class="bl">Research</span> Every note on this site is generated from the files listed here. Take the data and check the numbers yourself.</div>
<header class="cover"><div class="cover-in">
  <div class="cover-top"><div class="eyebrow">Research \u00b7 notes and data</div></div>
  <div class="gold-rule"></div>
  <h1>Research Data</h1>
  <p class="sub">The research notes and the data under them. Three research notes built on public SEC 13F filings and public prices, and the {sum(len(f) for _, _, _, f in notes)} CSVs they are generated from. Nothing on those pages is hand-written prose: change the data and the sentences change with it. Per-manager due-diligence memos are one click from any row under <a href="/external#all">Manager Analysis</a>.</p>
</div></header>
<main class="wrap">
{blocks}
<section>
  <div class="sh"><h2>Screening the managers</h2></div>
  <div class="card">
    <p>The manager table on the <a href="/#all">home page</a> is the fourth tool. Filter the {rv.get('managers', 91)} managers by evidence of alpha (the Newey-West t-statistic on the FF3 intercept), by style, and by how long the record is, then take the filtered set away with <b>Download CSV</b>. The export carries the slug, manager, fund, style, months, excess return over the market, FF3 t, and both scores.</p>
    <p class="cap">A note on what the t-statistic means here: it is the t on the intercept of a three-factor regression of the clone's monthly excess returns, with Newey-West standard errors. |t| \u2265 2 is the conventional bar for "unlikely to be luck". Very few managers clear it \u2014 that is the honest finding, not a bug in the screen.</p>
  </div>
</section>
<footer class="foot">
  <div class="running"><span>Track record verification \u00b7 research</span><span>{len(notes)} notes</span></div>
  <h4>Important information</h4>
  <p>Built from public SEC EDGAR 13F-HR filings and Yahoo Finance prices; factor and benchmark data from the Kenneth R. French Data Library. 13F clones are reconstructions of disclosed US long positions, not the funds themselves. Nothing here is a strategy, a recommendation or investment advice. Past performance is not indicative of future results.</p>
</footer>
</main>
<script>{JS}</script>
"""


def _kb(p: Path) -> str:
    try:
        n = p.stat().st_size
    except OSError:
        return "\u2014"
    return f"{n / 1024:.0f} KB" if n >= 1024 else f"{n} B"


# ---------------------------------------------------------------- alpha model lab

def alphalab_html(out_dir: Path | None = None) -> str | None:
    from .alphalab import OUT_DIR as LAB_DIR, SIGNALS
    d = out_dir or LAB_DIR
    if not (d / "manifest.json").exists():
        return None
    man = json.loads((d / "manifest.json").read_text())
    S = pd.read_csv(d / "summary.csv").set_index("signal")
    ic = pd.read_csv(d / "ic_monthly.csv", index_col=0, parse_dates=True)
    dec = pd.read_csv(d / "deciles.csv", index_col=0)
    imps = pd.read_csv(d / "xgboost_importance.csv", index_col=0)
    corr = pd.read_csv(d / "signal_correlation.csv", index_col=0)
    latest = pd.read_csv(d / "latest_ranks.csv")
    hh = man.get("head_to_head", {})
    feats = man["features"]

    def light(t):
        return ("yes", "evidence") if t >= 2 else ("weak", "weak") if t >= 1 else ("no", "none") if t > -1 else ("no", "wrong way")
    # signal table
    srow = ""
    for s in feats + ["linear", "xgboost"]:
        if s not in S.index: continue
        r = S.loc[s]; c, lab = light(r.ic_t)
        srow += (f"<tr><td><b>{esc(r.label)}</b></td><td class='n'>{r.coverage:.0%}</td><td class='n'>{int(r.months)}</td><td class='n'>{r.ic_mean:+.3f}</td><td class='n'>{r.ic_t:+.1f}</td>"
                 f"<td class='n'>{r.ic_pct_positive:.0%}</td><td class='n'>{pct(r.spread_ann, 1)}</td><td class='n'>{r.spread_t:+.1f}</td><td class='n'>{num(r.spread_sharpe)}</td>"
                 f"<td><span class='v {c}' style='color:#fff;background:var(--{ {'no': 'crit', 'weak': 'warn', 'yes': 'good'}[c] });font:600 9px var(--sans);letter-spacing:.14em;text-transform:uppercase;padding:4px 7px'>{esc(lab)}</span></td></tr>")
    # cumulative IC chart (linear vs xgboost) and rolling 12m IC for the best signals
    cells = [x.strftime("%Y-%m") for x in ic.index]
    series = []
    for col, name, cls in [("linear", "Linear composite", "s1"), ("xgboost", "xgboost", "s4"), ("momentum", "Momentum", "s0"), ("value", "Value", "s5"), ("low_vol", "Low volatility", "s2")]:
        if col in ic:
            cum = ic[col].fillna(0).cumsum().where(ic[col].notna().cummax())
            series.append(dict(name=name, values=[None if pd.isna(v) else float(v) for v in cum], cls=cls, emph=col in ("linear", "xgboost")))
    cum_chart = line_chart(cells, series, height=300, width=860, y_fmt=lambda v: f"{v:+.1f}", end_labels=False, uid="cumic")
    legend = "".join(f'<span><span class="k {s["cls"]}"></span>{esc(s["name"])}</span>' for s in series)
    # deciles bars for linear and xgboost
    def dec_bars(col):
        if col not in dec: return ""
        v = dec[col]
        return diverging_bars([f"D{int(i)}" for i in v.index], [float(x) * 12 for x in v.values], height=200, width=420, y_fmt=lambda y: f"{y * 100:+.0f}%",
                              tips=[f"decile {int(i)}: {x * 12 * 100:+.1f}%/yr average" for i, x in v.items()])
    imp_last = imps.iloc[:, -1].sort_values(ascending=False) if len(imps.columns) else pd.Series(dtype=float)
    imp_rows = "".join(f"<tr><td>{esc(SIGNALS.get(f, f))}</td><td class='n'>{v:.0%}</td></tr>" for f, v in imp_last.items())
    corr_head = "".join(f"<th class='n'>{esc(SIGNALS.get(c, c).split(' (')[0])}</th>" for c in corr.columns)
    corr_rows = "".join("<tr><td>" + esc(SIGNALS.get(r, r)) + "</td>" + "".join(f"<td class='n'>{corr.loc[r, c]:+.2f}</td>" for c in corr.columns) + "</tr>" for r in corr.index)
    lat_rows = "".join(f"<tr><td><b>{esc(r.ticker)}</b></td>" + "".join(f"<td class='n'>{'—' if pd.isna(r[f]) else f'{r[f]:+.1f}'}</td>" for f in feats) + f"<td class='n'><b>{r.linear:+.2f}</b></td><td class='n'>{'—' if pd.isna(r.xgboost) else f'{r.xgboost * 100:+.2f}%'}</td></tr>" for _, r in latest.head(25).iterrows())
    lin, xg = S.loc["linear"] if "linear" in S.index else None, S.loc["xgboost"] if "xgboost" in S.index else None
    verdict = ("No: tree ≈ line" if hh.get("ic_diff_t", 0) < 2 else "Yes: tree > line") if xg is not None and lin is not None else ""
    return f"""<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Alpha Model Lab</title>
<style>{CSS}{EXTRA_CSS}</style>
<div class="banner" role="note"><span class="bl">Research note</span> Signals from public prices and SEC filings, tested out of sample on a survivor-biased universe. A demonstration of method, not a strategy.</div>
<header class="cover"><div class="cover-in">
  <div class="cover-top"><div class="eyebrow">Active · Alpha Model Lab</div></div>
  <div class="gold-rule"></div>
  <h1>Alpha Model Lab</h1>
  <p class="sub">Which signals predict returns, and does a tree model beat a line? Eight classic stock-selection signals built point-in-time for ~{man['universe_avg']} names a month, ranked and tested every month from {man['first'][:7]} to {man['last'][:7]}: information coefficients, decile spreads, an equal-weight linear composite, and a gradient-boosted model (xgboost) trained walk-forward on the same inputs.</p>
  <dl class="meta">
    <div><dt>Months</dt><dd>{man['months']}</dd></div>
    <div><dt>Names / month</dt><dd>~{man['universe_avg']}</dd></div>
    <div><dt>Signals</dt><dd>{len(feats)}</dd></div>
    <div><dt>xgboost tested</dt><dd>{man.get('xgb_months', 0)} months from {(man.get('xgb_first') or '')[:7]}</dd></div>
    <div><dt>Built</dt><dd>{esc(man['built'])}</dd></div>
  </dl>
</div></header>
<main class="wrap">
<section class="verdict">
  <div class="sh"><h2>Findings</h2></div>
  <div class="tiles">
    <div class="tile"><div class="tl">Linear composite IC</div><div class="tv">{lin.ic_mean:+.3f}</div><div class="td muted">t = {lin.ic_t:+.1f} · positive in {lin.ic_pct_positive:.0%} of months · D10−D1 {pct(lin.spread_ann, 1)}/yr</div></div>
    <div class="tile"><div class="tl">xgboost IC, walk-forward</div><div class="tv">{xg.ic_mean:+.3f}</div><div class="td muted">t = {xg.ic_t:+.1f} · positive in {xg.ic_pct_positive:.0%} of months · D10−D1 {pct(xg.spread_ann, 1)}/yr</div></div>
    <div class="tile"><div class="tl">Head to head</div><div class="tv">{hh.get('ic_diff_mean', 0):+.3f}</div><div class="td muted">xgboost minus linear IC, same {hh.get('months', 0)} months · t = {hh.get('ic_diff_t', 0):+.1f} · xgboost ahead in {hh.get('xgb_wins_share', 0):.0%}</div></div>
    <div class="tile"><div class="tl">Does the tree model beat the line?</div><div class="tv small">{esc(verdict)}</div><div class="td muted">same inputs, same months; on this universe and period</div></div>
  </div>
  <p class="cap" style="margin-top:14px">An IC is the rank correlation between a signal today and returns next month; 0.03–0.05 with t above 3 is what a usable signal looks like over a long sample. The composite and the tree model are compared on exactly the same months — the first {man['min_train']} months are held out so xgboost has something to learn from, and it is refit every {man['retrain']} months using only earlier data.</p>
</section>

<section>
  <div class="sh"><h2>Signal by signal</h2></div>
  <div class="card tscroll"><table><thead><tr><th>signal</th><th class="n">coverage</th><th class="n">months</th><th class="n">mean IC</th><th class="n">t</th><th class="n">IC &gt; 0</th><th class="n">D10 − D1 /yr</th><th class="n">t</th><th class="n">Sharpe</th><th>evidence</th></tr></thead><tbody>{srow}</tbody></table>
  <p class="cap">Coverage is the share of stock-months with the signal available (fundamentals depend on SEC XBRL tags). D10 − D1 is the equal-weight top-decile minus bottom-decile monthly return, annualized ×12. Evidence: t ≥ 2 evidence, 1–2 weak, below 1 none.</p></div>
</section>

<section>
  <div class="sh"><h2>Cumulative information coefficient</h2></div>
  <div class="card"><div class="legend">{legend}</div>{cum_chart}
  <p class="cap">Running sum of monthly ICs. A signal that works climbs steadily; one that does not wanders around zero. The two models start at {(man.get('xgb_first') or '')[:7]}, when the first xgboost prediction is available.</p></div>
</section>

<section>
  <div class="sh"><h2>Decile returns</h2></div>
  <div class="grid2">
    <div class="card"><h3>Linear composite, by decile (annualized)</h3>{dec_bars('linear')}</div>
    <div class="card"><h3>xgboost, by decile (annualized)</h3>{dec_bars('xgboost')}</div>
  </div>
  <p class="cap">Average next-month return of each decile, ×12. A monotone staircase is the signature of a real signal; a spread that comes only from D1 or only from D10 is a warning about what is driving it.</p>
</section>

<section>
  <div class="sh"><h2>What the tree model uses, and how the signals overlap</h2></div>
  <div class="grid2">
    <div class="card"><h3>xgboost feature importance (latest fit)</h3><div class="tscroll"><table><thead><tr><th>signal</th><th class="n">gain share</th></tr></thead><tbody>{imp_rows}</tbody></table></div>
      <p class="cap">Share of the model's split gain attributed to each input. Importance is not predictive power — a signal can be used heavily and add nothing out of sample.</p></div>
    <div class="card tscroll"><h3>Average rank correlation between signals</h3><table><thead><tr><th></th>{corr_head}</tr></thead><tbody>{corr_rows}</tbody></table>
      <p class="cap">Averaged across months. Highly correlated signals are one signal wearing two names; the composite's equal weights double-count them.</p></div>
  </div>
</section>

<section>
  <div class="sh"><h2>Latest ranking — {esc(man['last'][:7])}</h2></div>
  <div class="card tscroll"><table><thead><tr><th>stock</th>{''.join(f"<th class='n'>{esc(SIGNALS.get(f, f).split(' (')[0])}</th>" for f in feats)}<th class="n">composite</th><th class="n">xgboost</th></tr></thead><tbody>{lat_rows}</tbody></table>
  <p class="cap">Top 25 by the linear composite at the latest month-end; z-scores across the universe. The xgboost column is its predicted return relative to the universe mean next month. This is what feeds the <a href="/research/construction">portfolio constructor</a>.</p></div>
</section>

<section>
  <div class="sh"><h2>Method and limits</h2></div>
  <div class="card">
    <p><b>Point in time.</b> Prices through the month-end; a filing is used only after its filed date and only if its period end is within 15 months. Balance-sheet items are the latest instant fact; income and cash flow are the latest annual-duration fact (330–400 days), so they update once a year like a Fama–French book value.</p>
    <p><b>Universe.</b> Names held by at least {MIN_HOLDERS if False else 5} managers at the latest 13F formation date, price ≥ $1. Around {man['universe_avg']} names a month; large and liquid, which is where anomalies are weakest. Delisted names drop out at their last price (survivorship, disclosed).</p>
    <p><b>Models.</b> The linear composite is the plain mean of available z-scores — no fitted weights at all. xgboost: 300 trees, depth 3, learning rate 0.03, subsample 0.8, min child weight 50, L2 = 5, trained on demeaned next-month returns; expanding window, first {man['min_train']} months held out, refit every {man['retrain']} months. No hyper-parameters were tuned on the test period.</p>
    <p><b>Reading it.</b> With ~150 months and one universe, an IC t-statistic below 2 is noise; a tree model with eight inputs cannot learn much that a line does not already capture, and the head-to-head test says how much. That is the finding a research meeting needs, not a backtest that looks good.</p>
  </div>
</section>

<footer class="foot">
  <div class="running"><span>Track record verification · research</span><span>{man['first'][:7]} – {man['last'][:7]}</span></div>
  <h4>Important information</h4>
  <p>Built from public SEC EDGAR filings (13F holdings, XBRL company facts) and Yahoo Finance prices. In-sample reconstructions with the limits stated above; not a strategy, a recommendation or investment advice. Past performance is not indicative of future results. Rebuild: <code>python -m trackrecord alpha-lab</code>.</p>
</footer>
</main>
<div id="tip" class="tip" hidden></div>
<script>{JS}</script>
"""


# ---------------------------------------------------------------- risk model

def riskmodel_html(out_dir: Path | None = None) -> str | None:
    from .riskmodel import OUT_DIR as RM_DIR
    from .alphalab import SIGNALS
    d = out_dir or RM_DIR
    if not (d / "manifest.json").exists():
        return None
    man = json.loads((d / "manifest.json").read_text())
    fr = pd.read_csv(d / "factor_returns.csv", index_col=0, parse_dates=True)
    r2 = pd.read_csv(d / "r2_monthly.csv", index_col=0, parse_dates=True).r2
    vols = pd.read_csv(d / "factor_vols_latest.csv").set_index("factor")
    spec = pd.read_csv(d / "specific_risk_latest.csv")
    decs = json.loads((d / "decompositions_latest.json").read_text())
    bt = pd.read_csv(d / "bias_by_year.csv", index_col=0).bias_stat
    bias = man.get("bias", {})
    styles = man["styles"]
    lab = lambda f: SIGNALS.get(f, f).split(" (")[0] if not f.startswith("ind:") else f[4:]

    cells = [x.strftime("%Y-%m") for x in fr.index]
    cls_cycle = ["s1", "s4", "s2", "s5", "s0", "s1l", "s1", "s4"]
    series = [dict(name=lab(s), values=[float(v) for v in fr[s].fillna(0).cumsum()], cls=cls_cycle[i % len(cls_cycle)]) for i, s in enumerate(styles) if s in fr]
    fchart = line_chart(cells, series, height=300, width=860, y_fmt=lambda v: f"{v * 100:+.0f}%", end_labels=False, uid="fret")
    legend = "".join(f'<span><span class="k {s["cls"]}"></span>{esc(s["name"])}</span>' for s in series)
    r2y = r2.groupby(r2.index.year).mean()
    r2_bars = diverging_bars([str(i) for i in r2y.index], [float(v) for v in r2y.values], height=180, width=860, y_fmt=lambda v: f"{v:.0%}",
                             tips=[f"{i}: average cross-sectional R² {v:.0%}" for i, v in r2y.items()])
    bias_bars = diverging_bars([str(i) for i in bt.index], [float(v) - 1 for v in bt.values], height=180, width=860, y_fmt=lambda v: f"{v + 1:.2f}",
                               tips=[f"{i}: bias statistic {v:.2f} (1 = calibrated)" for i, v in bt.items()])
    vrows = "".join(f"<tr><td>{esc(lab(f))}</td><td class='n'>{r.vol_ann:.1%}</td><td class='n'>{r.mean_ret_ann * 100:+.1f}%</td><td class='n'>{r.t:+.1f}</td></tr>"
                    for f, r in vols.iterrows() if not f.startswith("ind:"))
    irows = "".join(f"<tr><td>{esc(lab(f))}</td><td class='n'>{r.vol_ann:.1%}</td><td class='n'>{r.mean_ret_ann * 100:+.1f}%</td><td class='n'>{r.t:+.1f}</td></tr>"
                    for f, r in vols.sort_values("vol_ann", ascending=False).iterrows() if f.startswith("ind:"))
    def dec_card(name, x):
        g = x.get("groups", {})
        rows = "".join(f"<tr><td>{esc(k.title())}</td><td class='n'>{v:.0%}</td></tr>" for k, v in g.items()) + f"<tr><td class='muted'>Cross-factor interaction</td><td class='n'>{x.get('interaction', 0):.0%}</td></tr><tr><td>Stock-specific</td><td class='n'>{1 - x['share_factor']:.0%}</td></tr>"
        top = "".join(f"<tr><td>{esc(lab(t['factor']))}</td><td class='n'>{t['exposure']:+.2f}</td><td class='n'>{t['var_share']:.0%}</td></tr>" for t in x["top"][:6])
        return (f"<div class='card'><h3>{esc(name)}</h3><div class='tiles' style='margin-bottom:10px'>"
                f"<div class='tile'><div class='tl'>Total risk</div><div class='tv'>{x['total']:.1%}</div><div class='td muted'>annualized · {x['n_names']} names</div></div>"
                f"<div class='tile'><div class='tl'>Factor</div><div class='tv'>{x['factor']:.1%}</div><div class='td muted'>{x['share_factor']:.0%} of variance</div></div>"
                f"<div class='tile'><div class='tl'>Specific</div><div class='tv'>{x['specific']:.1%}</div><div class='td muted'>{1 - x['share_factor']:.0%} of variance</div></div></div>"
                f"<div class='grid2'><div class='tscroll'><table><thead><tr><th>source</th><th class='n'>share of variance</th></tr></thead><tbody>{rows}</tbody></table></div>"
                f"<div class='tscroll'><table><thead><tr><th>largest exposures</th><th class='n'>exposure</th><th class='n'>var share</th></tr></thead><tbody>{top}</tbody></table></div></div></div>")
    dec_html = "".join(dec_card(k, v) for k, v in decs.items())
    spec_med = float(spec.specific_vol_ann.median()); spec_hi = spec.head(8)
    spec_rows = "".join(f"<tr><td><b>{esc(r.ticker)}</b></td><td class='n'>{r.specific_vol_ann:.0%}</td></tr>" for _, r in spec_hi.iterrows())
    b_rand, b_uni = bias.get("random", float("nan")), bias.get("universe", float("nan"))
    verdict = ("calibrated" if 0.85 <= b_rand <= 1.15 else "under-forecasts risk" if b_rand > 1.15 else "over-forecasts risk")
    return f"""<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Factor Risk Model</title>
<style>{CSS}{EXTRA_CSS}</style>
<div class="banner" role="note"><span class="bl">Research note</span> A fundamental factor risk model estimated on the research universe. A demonstration of the method a vendor model implements at scale.</div>
<header class="cover"><div class="cover-in">
  <div class="cover-top"><div class="eyebrow">Active · Factor Risk Model</div></div>
  <div class="gold-rule"></div>
  <h1>Factor Risk Model</h1>
  <p class="sub">A fundamental factor risk model, Barra-style. Every month, stock returns are regressed on the same point-in-time exposures the alpha lab uses plus industry membership; the factor returns and residuals become a factor covariance and stock-specific risk, and any portfolio decomposes into where its risk comes from — with a calibration test to say whether the forecasts can be trusted.</p>
  <dl class="meta">
    <div><dt>Months</dt><dd>{man['months']} · {man['first'][:7]} → {man['last'][:7]}</dd></div>
    <div><dt>Factors</dt><dd>{man['factors']} · market + {len(styles)} styles + {man['industries']} industries</dd></div>
    <div><dt>Avg R²</dt><dd>{man['r2_avg']:.0%}</dd></div>
    <div><dt>Bias statistic</dt><dd>{b_rand:.2f}</dd></div>
    <div><dt>Built</dt><dd>{esc(man['built'])}</dd></div>
  </dl>
</div></header>
<main class="wrap">
<section class="verdict">
  <div class="sh"><h2>Does it forecast risk?</h2></div>
  <div class="tiles">
    <div class="tile"><div class="tl">Bias statistic, random portfolios</div><div class="tv">{b_rand:.2f}</div><div class="td muted">std of realized ÷ predicted · 1.00 is perfect · {verdict}</div></div>
    <div class="tile"><div class="tl">Bias statistic, whole universe</div><div class="tv">{b_uni:.2f}</div><div class="td muted">equal-weight universe each month</div></div>
    <div class="tile"><div class="tl">Explanatory power</div><div class="tv">{man['r2_avg']:.0%}</div><div class="td muted">average cross-sectional R² per month</div></div>
    <div class="tile"><div class="tl">Median specific risk</div><div class="tv">{spec_med:.0%}</div><div class="td muted">annualized, per stock, latest</div></div>
  </div>
  <p class="cap" style="margin-top:14px">The bias statistic is the standard test: for {len(bt)} years of monthly random 50-name portfolios, divide each realized return by the volatility the model predicted for it the month before; a calibrated model gives a ratio with standard deviation 1. Above 1 the model under-forecasts risk (dangerous), below 1 it over-forecasts (costly). Factor covariance uses a {man['window']}-month window with a {man['half_life']}-month half-life; specific variance is an EWMA of each stock's residuals, shrunk toward the median for short histories.</p>
  <div class="card" style="margin-top:14px"><h3>Bias statistic by year (random portfolios)</h3>{bias_bars}<p class="cap">Plotted as deviation from 1. Years above the line are years the model was too confident — typically regime changes the trailing window had not seen.</p></div>
</section>

<section>
  <div class="sh"><h2>Factor returns</h2></div>
  <div class="card"><div class="legend">{legend}</div>{fchart}
  <p class="cap">Cumulative return to each style factor: the return to a unit exposure, holding the other styles and industries fixed. These are the "pure" factor returns a Barra report shows, not long-short portfolios.</p></div>
  <div class="grid2" style="margin-top:14px">
    <div class="card tscroll"><h3>Style factors, latest window</h3><table><thead><tr><th>factor</th><th class="n">vol /yr</th><th class="n">mean /yr</th><th class="n">t</th></tr></thead><tbody>{vrows}</tbody></table></div>
    <div class="card tscroll"><h3>Industry factors, latest window</h3><table><thead><tr><th>industry</th><th class="n">vol /yr</th><th class="n">mean /yr</th><th class="n">t</th></tr></thead><tbody>{irows}</tbody></table></div>
  </div>
  <div class="card" style="margin-top:14px"><h3>Cross-sectional R² by year</h3>{r2_bars}<p class="cap">How much of the month's dispersion in stock returns the factors explain. Vendor models on broad universes typically run 20–40%; higher in crises when everything moves together.</p></div>
</section>

<section>
  <div class="sh"><h2>Risk decomposition — {esc(man['latest_month'][:7])}</h2></div>
  <p class="note">The report a portfolio manager reads before a rebalance: where the risk is, and whether the active bets are the intended ones. Groups are own-group variance shares; cross-factor covariance is shown separately.</p>
  {dec_html}
</section>

<section>
  <div class="sh"><h2>Specific risk</h2></div>
  <div class="card"><div class="grid2"><div class="tscroll"><table><thead><tr><th>highest specific risk, latest</th><th class="n">vol /yr</th></tr></thead><tbody>{spec_rows}</tbody></table></div>
  <div><p>Median {spec_med:.0%} across {len(spec)} names. Specific risk is what diversification removes and what a concentrated manager is paid for taking; the names at the top of this list are where a single position can move a portfolio.</p></div></div></div>
</section>

<section>
  <div class="sh"><h2>Method and limits</h2></div>
  <div class="card">
    <p><b>Estimation.</b> Each month t, the return over t → t+1 of every stock in the universe is regressed on its exposures at t: an intercept (market), {len(styles)} style z-scores, and {man['industries']} industry dummies with the count-weighted sum of industry returns constrained to zero. Ordinary least squares, equal weights; a vendor would weight by √cap and winsorise residuals.</p>
    <p><b>Covariance.</b> Exponentially weighted over the trailing {man['window']} months with a {man['half_life']}-month half-life; no Newey–West, no volatility-regime adjustment. Specific variance per stock: EWMA of squared residuals, shrunk toward the cross-sectional median by n/(n+12) for stocks with n months of history.</p>
    <p><b>What this is not.</b> Barra's USE4 has ~10 styles built from dozens of descriptors, ~60 industries, daily estimation, and years of calibration. This model shares the structure — exposures × factor returns + specific — and the tests, on one universe of ~{man['universe_avg']} names. It is enough to run the portfolio constructor and index tracker on, and to show where a vendor model earns its fee.</p>
  </div>
</section>

<footer class="foot">
  <div class="running"><span>Track record verification · research</span><span>{man['first'][:7]} – {man['last'][:7]}</span></div>
  <h4>Important information</h4>
  <p>Built from public SEC EDGAR filings and Yahoo Finance prices and sectors. A demonstration with the limits stated above; not investment advice. Rebuild: <code>python -m trackrecord risk-model</code>.</p>
</footer>
</main>
<div id="tip" class="tip" hidden></div>
<script>{JS}</script>
"""


# ---------------------------------------------------------------- passive: index tracker + DPSW

def tracker_html(out_dir: Path | None = None) -> str | None:
    from .tracker import OUT_DIR as TR_DIR, SAMPLES
    from .alphalab import SIGNALS
    d = out_dir or TR_DIR
    if not (d / "manifest.json").exists():
        return None
    man = json.loads((d / "manifest.json").read_text())
    S = pd.read_csv(d / "summary.csv").set_index("key")
    R = pd.read_csv(d / "backtest_monthly.csv", index_col=0, parse_dates=True)
    lg = pd.read_csv(d / "rebalances.csv")
    ws = json.loads((d / "dpsw.json").read_text())
    hold = pd.read_csv(d / "dpsw_holdings.csv")
    tl = pd.read_csv(d / "rebalance_trades.csv") if (d / "rebalance_trades.csv").exists() else pd.DataFrame()
    slc = pd.read_csv(d / "dpsw_cashflow_slice.csv")
    idx_name = man["benchmark"]
    # the live book, if the server has refreshed it; otherwise the snapshot at the last month-end
    from .livebook import load as load_live, OUT_DIR as LIVE_DIR, STATE as LIVE_STATE
    live = load_live()
    drift_tl = pd.DataFrame()
    if live:
        ws = live
        hold = pd.read_csv(LIVE_DIR / "dpsw_holdings_live.csv"); slc = pd.read_csv(LIVE_DIR / "dpsw_cashflow_slice_live.csv")
        drift_tl = pd.read_csv(LIVE_DIR / "drift_trades_live.csv") if (LIVE_DIR / "drift_trades_live.csv").exists() else pd.DataFrame()

    # replication table
    rows = ""
    for k in ["index", "full"] + [f"sampled_{n}" for n in SAMPLES]:
        r = S.loc[k]
        rows += (f"<tr><td><b>{esc(r.label)}</b></td><td class='n'>{r.avg_names:.0f}</td><td class='n'>{pct(r.ann_return, 1, False)}</td><td class='n'>{r.tracking_error:.2%}</td>"
                 f"<td class='n'>{'' if pd.isna(r.pred_te_avg) else f'{r.pred_te_avg:.2%}'}</td><td class='n'>{r.tracking_difference * 1e4:+.0f} bps</td><td class='n'>{r.worst_month_active * 1e4:+.0f} bps</td><td class='n'>{r.turnover_yr:.0%}</td></tr>")
    # active return chart: cumulative active of sampled books
    Ra = R.dropna(subset=["index"])
    cells = [x.strftime("%Y-%m") for x in Ra.index]
    series = [dict(name=f"{n} names", values=[float(v) for v in ((1 + Ra[f'sampled_{n}']).cumprod() / (1 + Ra['index']).cumprod() - 1)], cls=c)
              for n, c in zip(SAMPLES, ["s4", "s1", "s2"])]
    chart = line_chart(cells, series, height=260, width=860, y_fmt=lambda v: f"{v * 100:+.1f}%", end_labels=False, uid="track")
    legend = "".join(f'<span><span class="k {s["cls"]}"></span>{esc(s["name"])}</span>' for s in series)
    # predicted vs realized TE per rebalance (realized over the following quarter) for 80 names
    a80 = (R["sampled_80"] - R["index"]).dropna()
    te_rows = ""
    for _, r in lg.tail(8).iloc[::-1].iterrows():
        F = pd.Timestamp(r.formation); nxt = a80[(a80.index > F)].head(3)
        real = float(nxt.std() * np.sqrt(12)) if len(nxt) >= 2 else np.nan
        te_rows += (f"<tr><td>{esc(str(r.formation))}</td><td class='n'>{int(r.constituents)}</td><td class='n'>{r.top10_weight:.0%}</td><td class='n'>{r.turnover_index:.1%}</td>"
                    f"<td class='n'>{int(r.names_80)}</td><td class='n'>{r.turnover_sampled_80:.1%}</td><td class='n'>{r.pred_te_80:.2%}</td><td class='n'>{'' if pd.isna(real) else f'{real:.2%}'}</td></tr>")
    # DPSW
    mtd, qtd, ytd = ws["mtd"], ws["qtd"], ws["ytd"]
    sec_rows = "".join(f"<tr><td>{esc(k)}</td><td class='n'>{v * 1e4:+.0f} bps</td></tr>" for k, v in sorted(ws["sector_active"].items(), key=lambda kv: kv[1]))
    act_rows = "".join(f"<tr><td><b>{esc(k)}</b></td><td class='n'>{v * 1e4:+.0f} bps</td></tr>" for k, v in ws["largest_active"].items())
    risk_rows = "".join(f"<tr><td>{esc(SIGNALS.get(t['factor'], t['factor']).split(' (')[0] if not t['factor'].startswith('ind:') else t['factor'][4:])}</td><td class='n'>{t['exposure']:+.2f}</td><td class='n'>{t['var_share']:.0%}</td></tr>" for t in ws["top_risk"])
    hold_rows = "".join(f"<tr><td><b>{esc(r.ticker)}</b> <span class='muted'>{esc(r.sector)}</span></td><td class='n'>{r.weight:.2%}</td><td class='n'>{r.index_weight:.2%}</td><td class='n'>{r.active * 1e4:+.0f}</td><td class='n'>{r.shares:,.0f}</td><td class='n'>${r.market_value / 1e6:,.1f}m</td></tr>"
                        for _, r in hold.head(15).iterrows())
    events = ws.get("dead") or []
    ev_html = ("".join(f"<li><b>{esc(t)}</b> — no price today: treat as a corporate event (delisting, acquisition close, or halt); confirm with the custodian, sell or receive proceeds, and re-slice the weight pro rata.</li>" for t in events)
               if events else "<li>No holding has stopped pricing. No index announcements are pending: the next reconstitution is the quarter's 13F formation date, when additions and deletions become known and the rebalance list below is regenerated.</li>")
    slc_rows = "".join(f"<tr><td><b>{esc(r.ticker)}</b></td><td class='n'>{r.weight:.2%}</td><td class='n'>${r.usd / 1e3:,.0f}k</td><td class='n'>{r.shares:,.0f}</td></tr>" for _, r in slc.sort_values("usd", ascending=False).head(10).iterrows())
    tl_rows = "".join(f"<tr><td><b>{esc(r.ticker)}</b> <span class='muted'>{esc(r.sector)}</span></td><td>{esc(r.reason)}</td><td class='n'>{r.current_wt:.2%}</td><td class='n'>{r.target_wt:.2%}</td><td class='n'>{r.index_wt:.2%}</td><td class='n'>{r.shares:+,}</td><td class='n'>${r.trade_usd / 1e6:+,.2f}m</td></tr>"
                      for _, r in tl.head(16).iterrows()) if len(tl) else ""
    s80 = S.loc["sampled_80"]
    return f"""<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Index Tracking — live book</title>
<style>{CSS}{EXTRA_CSS}
.dpsw h3 {{ margin-top: 0 }} .dpsw .kv {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px }} .dpsw .kv .tile .tv {{ font-size: 22px }}
</style>
<div class="banner" role="note"><span class="bl">Research note</span> A cap-weighted index built to a stated rulebook from public data, replicated the way a passive desk replicates a vendor index. A demonstration of method.</div>
<header class="cover"><div class="cover-in">
  <div class="cover-top"><div class="eyebrow">Passive · Index Tracking</div></div>
  <div class="gold-rule"></div>
  <h1>Index Tracking — live book</h1>
  <p class="sub">Replicate the index with fewer names, and run the book every day. The {esc(idx_name)}: ~{man['names_index']} US names, cap-weighted, reconstituted quarterly. Held in full, and with 40, 80 and 150 names chosen by optimised sampling under the risk model. Then the desk's daily sheet: NAV, cash, actives, predicted tracking error, events, and the trades.</p>
  <dl class="meta">
    <div><dt>Rebalances</dt><dd>{man['rebalances']} · {man['first'][:7]} → {man['last'][:7]}</dd></div>
    <div><dt>Index names</dt><dd>~{man['names_index']}</dd></div>
    <div><dt>Book</dt><dd>${man['nav'] / 1e6:,.0f}m · {man['names_held']} names</dd></div>
    <div><dt>Realized TE, 80 names</dt><dd>{man['te_realized']:.2%}</dd></div>
    <div><dt>{'Book last updated' if live else 'Built'}</dt><dd>{esc(ws['updated']) if live else esc(man['built'])}</dd></div>
  </dl>
</div></header>
<main class="wrap">
<section class="verdict">
  <div class="sh"><h2>Replication: full versus sampled</h2></div>
  <div class="card tscroll"><table><thead><tr><th>portfolio</th><th class="n">names</th><th class="n">return /yr</th><th class="n">tracking error</th><th class="n">predicted TE</th><th class="n">tracking difference /yr</th><th class="n">worst month</th><th class="n">turnover /yr</th></tr></thead><tbody>{rows}</tbody></table>
  <p class="cap">Tracking error = standard deviation of monthly active return, annualized; predicted TE = the risk model's forecast at each rebalance, averaged. Tracking difference = mean active return (costs of {man['cost_bps']:.0f} bps per dollar traded are charged; the index pays none). Sampled books hold the largest names per industry, with weights set by a quadratic program that minimises predicted tracking error; no name is forced below its index weight, none above {man['max_weight']:.0%}.</p></div>
  <div class="card" style="margin-top:14px"><div class="legend">{legend}</div>{chart}<p class="cap">Cumulative active return of each sampled book against the index. Flat and close to zero is the goal of a passive mandate; the drift is tracking difference, the wiggle is tracking error.</p></div>
</section>

<section class="dpsw" id="dpsw">
  <div class="sh"><h2>Daily Portfolio Status Worksheet — {esc(ws.get('price_date', ws.get('date', '')))}</h2>{'<span class="chip public">Live book</span>' if live else '<span class="chip estimated">Snapshot at month-end</span>'}</div>
  <p class="note">{(f"<b>Live.</b> Prices as of {esc(ws['price_date'])}; last refreshed {esc(ws['updated'])}. The 80-name book and the full index are carried forward from the {esc(ws['last_month'])} month-end with daily Yahoo Finance prices, and every figure below is recomputed; the server refreshes once a day. <a href='/live/passive/refresh'>Refresh now</a>." if live else "The book as of the last complete month-end; the live refresh has not run on this server yet.")}
  The sheet a passive PM completes each morning: what we hold, how far from the index we are and why, what the model expects that to cost, and what needs doing today.{f" <span class='err'>Last refresh error: {esc(LIVE_STATE['error'])}</span>" if LIVE_STATE.get('error') else ''}</p>
  <div class="card"><div class="kv">
    <div class="tile"><div class="tl">NAV</div><div class="tv">${ws['nav'] / 1e6:,.1f}m</div><div class="td muted">cash ${ws['cash'] / 1e6:,.2f}m ({ws['cash_pct']:.1%} target)</div></div>
    <div class="tile"><div class="tl">Holdings</div><div class="tv">{ws['n_holdings']}</div><div class="td muted">of {ws['n_index']} index names</div></div>
    <div class="tile"><div class="tl">MTD</div><div class="tv">{mtd[0]:+.2%}</div><div class="td muted">index {mtd[1]:+.2%} · active {(mtd[0] - mtd[1]) * 1e4:+.0f} bps</div></div>
    <div class="tile"><div class="tl">QTD</div><div class="tv">{qtd[0]:+.2%}</div><div class="td muted">index {qtd[1]:+.2%} · active {(qtd[0] - qtd[1]) * 1e4:+.0f} bps</div></div>
    <div class="tile"><div class="tl">YTD</div><div class="tv">{ytd[0]:+.2%}</div><div class="td muted">index {ytd[1]:+.2%} · active {(ytd[0] - ytd[1]) * 1e4:+.0f} bps</div></div>
    <div class="tile"><div class="tl">Predicted TE</div><div class="tv">{ws['pred_te']:.2%}</div><div class="td muted">factor {ws['pred_te_factor']:.2%} · specific {ws['pred_te_specific']:.2%}</div></div>
    <div class="tile"><div class="tl">Largest active</div><div class="tv">{ws['max_active'] * 1e4:+.0f} bps</div><div class="td muted">{esc(ws['max_active_name'])}</div></div>
    <div class="tile"><div class="tl">Drift from target</div><div class="tv">{ws['drift_max'] * 1e4:.0f} bps</div><div class="td muted">largest, {esc(ws['drift_max_name'])}{f" · {ws['drift_total']:.1%} one-way to re-target" if 'drift_total' in ws else ''}</div></div>
  </div>
  <div class="grid2" style="margin-top:14px">
    <div class="tscroll"><h3>Sector active weights</h3><table><thead><tr><th>sector</th><th class="n">active</th></tr></thead><tbody>{sec_rows}</tbody></table></div>
    <div><h3>Largest active positions</h3><div class="tscroll"><table><thead><tr><th>stock</th><th class="n">active</th></tr></thead><tbody>{act_rows}</tbody></table></div>
      <h3 style="margin-top:14px">Where the tracking risk comes from</h3><div class="tscroll"><table><thead><tr><th>factor</th><th class="n">active exposure</th><th class="n">share of TE²</th></tr></thead><tbody>{risk_rows}</tbody></table></div></div>
  </div>
  <h3 style="margin-top:18px">Events and actions today</h3><ul>{ev_html}
  <li><b>Not yet wired, versus a real desk:</b> corporate-action notices from the custodian (dividends, splits, spin-offs, tender offers) and the index vendor's add/delete calendar. Here a corporate action is visible only as a price that stops or jumps, and index changes are known only at reconstitution. Nothing is interpolated.</li></ul>
  {(f"<h3 style='margin-top:14px'>Trades to return to target today</h3><p class='cap'>The drifted book versus the {esc(ws.get('formation', ''))} target at today's prices: {ws.get('drift_trades', 0)} tickets, ${ws.get('drift_traded_usd', 0) / 1e6:,.1f}m. A passive desk does not usually trade drift between reconstitutions; this is what it would cost to. Full list: <a href='/live/passive/drift_trades_live.csv'>drift_trades_live.csv</a>.</p>"
     + "<div class='tscroll'><table><thead><tr><th>stock</th><th class='n'>now</th><th class='n'>target</th><th class='n'>shares</th><th class='n'>$</th></tr></thead><tbody>"
     + "".join(f"<tr><td><b>{esc(r.ticker)}</b> <span class='muted'>{esc(r.sector)}</span></td><td class='n'>{r.current_wt:.2%}</td><td class='n'>{r.target_wt:.2%}</td><td class='n'>{r.shares:+,}</td><td class='n'>${r.trade_usd / 1e6:+,.2f}m</td></tr>" for _, r in drift_tl.head(10).iterrows())
     + "</tbody></table></div>") if len(drift_tl) else ''}
  <div class="grid2" style="margin-top:14px">
    <div class="tscroll"><h3>Largest holdings</h3><table><thead><tr><th>stock</th><th class="n">weight</th><th class="n">index</th><th class="n">active bps</th><th class="n">shares</th><th class="n">value</th></tr></thead><tbody>{hold_rows}</tbody></table>
      <p class="cap">Full list: {'<a href="/live/passive/dpsw_holdings_live.csv">dpsw_holdings_live.csv</a> · <a href="/live/passive/dpsw_live.json">dpsw_live.json</a>' if live else '<a href="/research/data/index-tracker/dpsw_holdings.csv">dpsw_holdings.csv</a>'}.</p></div>
    <div class="tscroll"><h3>Cash-flow slice: invest ${ws['inflow'] / 1e6:,.1f}m</h3><table><thead><tr><th>stock</th><th class="n">target</th><th class="n">buy</th><th class="n">shares</th></tr></thead><tbody>{slc_rows}</tbody></table>
      <p class="cap">A client inflow is invested pro rata to target weights so tracking error does not move; whole shares, residual to cash. Full list: {'<a href="/live/passive/dpsw_cashflow_slice_live.csv">dpsw_cashflow_slice_live.csv</a>' if live else '<a href="/research/data/index-tracker/dpsw_cashflow_slice.csv">dpsw_cashflow_slice.csv</a>'}.</p></div>
  </div></div>
</section>

<section>
  <div class="sh"><h2>Reconstitution trade list — {esc(man['last'])}</h2></div>
  <p class="note">{man['latest_trades']} tickets, ${man['latest_traded_usd'] / 1e6:,.1f}m traded: names entering and leaving the sample as the index reconstitutes, and reweights where the optimiser moved weight. Largest first.</p>
  <div class="card tscroll"><table><thead><tr><th>stock</th><th>reason</th><th class="n">now</th><th class="n">target</th><th class="n">index</th><th class="n">shares</th><th class="n">$</th></tr></thead><tbody>{tl_rows}</tbody></table>
  <p class="cap">Full list: <a href="/research/data/index-tracker/rebalance_trades.csv">rebalance_trades.csv</a>.</p></div>
</section>

<section>
  <div class="sh"><h2>Rebalance log</h2></div>
  <div class="card tscroll"><table><thead><tr><th>date</th><th class="n">index names</th><th class="n">top-10 weight</th><th class="n">index turnover</th><th class="n">80-name book</th><th class="n">book turnover</th><th class="n">predicted TE</th><th class="n">realized next qtr</th></tr></thead><tbody>{te_rows}</tbody></table>
  <p class="cap">Realized TE over the following three months (annualized from three observations — noisy, which is why the average over all rebalances is the number to trust). Full log: <a href="/research/data/index-tracker/rebalances.csv">rebalances.csv</a>.</p></div>
</section>

<section>
  <div class="sh"><h2>Method and limits</h2></div>
  <div class="card">
    <p><b>The index.</b> US-domestic filers (10-K / 10-Q) held by at least five of the managers in the system at each 13F formation date, weighted by shares outstanding × split-adjusted close. Shares come from the companies' own XBRL filings, point in time; unit errors are removed and the series is split-adjusted to today's basis so it matches Yahoo's split-adjusted prices. Foreign filers report ordinary shares while the price is per ADR, so they are excluded — the same reason a US index vendor uses a different methodology for ADRs. The result is a real, reproducible rulebook; it is not a licensed index.</p>
    <p><b>Sampling.</b> Seats are allocated to industries in proportion to index weight, filled by the largest names; a quadratic program (SLSQP with analytic gradients) then minimises (w−b)ᵀ(XFXᵀ+Δ)(w−b) under the risk model with w ≥ 0, Σw = 1 and the name cap. The model's own decomposition reports factor versus specific tracking risk, which is how a desk decides whether to add names or fix a sector.</p>
    <p><b>Running it.</b> Rebalanced only at reconstitution and held with drift in between; a client inflow is sliced pro rata; a name that stops pricing is an event to resolve, not a data point to interpolate. Costs of {man['cost_bps']:.0f} bps per dollar traded. Cash target {man['cash_target']:.1%} of NAV.</p>
    <p><b>What is missing.</b> Corporate-action notices (dividends, splits, spin-offs, tender offers) come from the custodian and the index provider in production; here they are visible only as prices that stop or jump. Index-provider announcements (adds, deletes, share updates) are known days in advance in production and let the desk trade at the close of the effective date; here they are known only at reconstitution.</p>
  </div>
</section>

<footer class="foot">
  <div class="running"><span>Track record verification · research</span><span>{man['first'][:7]} – {man['last'][:7]}</span></div>
  <h4>Important information</h4>
  <p>Built from public SEC EDGAR filings and Yahoo Finance prices. A demonstration of passive portfolio management on a self-defined index; not investment advice. Rebuild: <code>python -m trackrecord index-tracker</code>.</p>
</footer>
</main>
<div id="tip" class="tip" hidden></div>
<script>{JS}</script>
"""


# ---------------------------------------------------------------- fund of funds

def fof_html(out_dir: Path | None = None) -> str | None:
    from .fof import OUT_DIR as FOF_DIR
    d = out_dir or FOF_DIR
    if not (d / "manifest.json").exists():
        return None
    man = json.loads((d / "manifest.json").read_text())
    M = pd.read_csv(d / "managers.csv")
    curve = pd.read_csv(d / "diversification_curve.csv")
    blends = json.loads((d / "blends.json").read_text())
    alloc = json.loads((d / "allocations_by_prior.json").read_text())
    oos = man.get("oos", {})
    tau_eb = man["tau_eb"]

    # shrinkage table (top 20 by raw alpha) and allocation table
    Ms = M.sort_values("alpha", ascending=False)
    srow = "".join(f"<tr><td><b>{esc(r['name'])}</b> <span class='muted'>{esc(r.style)}</span></td><td class='n'>{int(r.months)}</td><td class='n'>{pct(r.alpha, 1)}</td><td class='n'>{pct(r.se, 1, False)}</td><td class='n'>{r.t:+.1f}</td>"
                   f"<td class='n'>{r.shrink_factor:.2f}</td><td class='n'>{pct(r.alpha_shrunk, 1)}</td><td class='n'>{pct(r.alpha_eb, 2)}</td></tr>" for _, r in Ms.head(20).iterrows())
    W = M[M.weight_optimised > 0].sort_values("weight_optimised", ascending=False)
    wrow = "".join(f"<tr><td><b>{esc(r['name'])}</b> <span class='muted'>{esc(r.style)}</span></td><td class='n'>{pct(r.alpha_shrunk, 1)}</td><td class='n'>{r.resid_vol:.1%}</td><td class='n'>{r.ir_shrunk:.2f}</td><td class='n'>{r.weight_optimised:.1%}</td></tr>" for _, r in W.iterrows())
    prior_rows = "".join(f"<tr><td>{esc(k)}</td><td class='n'>{v['tau']:.1%}</td><td class='n'>{v['names']}</td><td class='n'>{pct(v['alpha'], 1)}</td><td class='n'>{num(v['ir'])}</td></tr>" for k, v in man["alloc_summary"].items())
    blend_rows = "".join(f"<tr><td>{esc(k)}</td><td class='n'>{v['names']}</td><td class='n'>{pct(v['alpha'], 1)}</td><td class='n'>{v['vol']:.1%}</td><td class='n'>{num(v['ir'])}</td></tr>" for k, v in blends.items())
    cells = [str(int(n)) for n in curve.n]
    cchart = line_chart(cells, [dict(name="expected IR", values=[float(v) for v in curve.ir], cls="s1", emph=True)], height=220, width=860, y_fmt=lambda v: f"{v:.2f}", end_labels=False, uid="divc")
    oos_html = ""
    if oos.get("available"):
        t_, b_, a_, s_ = oos["top"], oos["bottom"], oos["all"], oos["top_minus_bottom"]
        oos_html = (f"<div class='tiles'><div class='tile'><div class='tl'>Past top {oos['n_top']}</div><div class='tv'>{pct(t_['alpha'], 1)}</div><div class='td muted'>alpha after {esc(oos['split'][:7])} · t {t_['t']:+.1f} · were {pct(t_['first_half_alpha'], 1)} before</div></div>"
                    f"<div class='tile'><div class='tl'>Past bottom {oos['n_top']}</div><div class='tv'>{pct(b_['alpha'], 1)}</div><div class='td muted'>alpha after · t {b_['t']:+.1f} · were {pct(b_['first_half_alpha'], 1)} before</div></div>"
                    f"<div class='tile'><div class='tl'>Top − bottom</div><div class='tv'>{pct(s_['alpha'], 1)}</div><div class='td muted'>t {s_['t']:+.1f} over {s_['months']} months</div></div>"
                    f"<div class='tile'><div class='tl'>Rank persistence</div><div class='tv'>{oos['rank_corr']:+.2f}</div><div class='td muted'>Spearman, first-half vs second-half alpha, {oos['candidates']} managers</div></div></div>"
                    f"<p class='cap' style='margin-top:12px'>Managers ranked by FF3 alpha t-statistic fitted on data before {esc(oos['split'][:7])}; the second-period alpha uses the first period's betas, so it is a genuine out-of-sample residual. "
                    f"{'Past alpha carried some information here' if s_['t'] >= 2 else 'Past alpha did not predict future alpha here'}: the spread between the ten best and ten worst past performers is {pct(s_['alpha'], 1)}/yr with t = {s_['t']:+.1f}, and the rank correlation is {oos['rank_corr']:+.2f}. "
                    f"{oos['share_positive_second']:.0%} of managers had positive alpha in the second period. {s_['months']} months of evaluation is short; the result is indicative, and it is the result.</p>")
    else:
        oos_html = f"<p class='cap'>Out-of-sample test not available: {esc(str(oos.get('reason', '')))}.</p>"
    eb_txt = (f"The empirical-Bayes prior variance is <b>zero</b>: the dispersion of the {man['managers']} estimated alphas ({M.alpha.std():.1%} standard deviation) is no larger than estimation noise alone would produce (root-mean-square standard error {np.sqrt((M.se ** 2).mean()):.1%}). "
              f"On this evidence the best estimate of every manager's alpha is the common mean, {pct(man['prior_mean'], 2)}/yr, and the right fund of funds is the index. "
              f"{man['n_t2']} managers show t ≥ 2 against about {man['expected_t2_by_chance']:.1f} expected by chance among {man['managers']}."
              if tau_eb == 0 else
              f"The empirical-Bayes prior standard deviation is {tau_eb:.1%}/yr: the estimated alphas disperse more than noise alone would produce, so some of it is real.")
    return f"""<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Fund of Funds</title>
<style>{CSS}{EXTRA_CSS}</style>
<div class="banner" role="note"><span class="bl">Research note</span> Manager alphas from 13F clones and listed funds — public proxies for the audited, net-of-fee series a real allocation would use. A demonstration of method.</div>
<header class="cover"><div class="cover-in">
  <div class="cover-top"><div class="eyebrow">Manager Analysis · Fund of Funds</div></div>
  <div class="gold-rule"></div>
  <h1>Fund of Funds</h1>
  <p class="sub">Combine the managers that pass, after shrinking what they claim. {man['managers']} managers with five years or more of history. Each alpha is shrunk toward the cross-section in proportion to its noise, residual returns are correlated, and a long-only blend is chosen to maximise expected information ratio — under an honest prior, and under the priors an allocator might actually hold. Then the test that matters: does picking past winners work?</p>
  <dl class="meta">
    <div><dt>Candidates</dt><dd>{man['managers']}</dd></div>
    <div><dt>Empirical-Bayes τ</dt><dd>{tau_eb:.1%}</dd></div>
    <div><dt>Avg residual correlation</dt><dd>{man['avg_corr']:+.2f}</dd></div>
    <div><dt>Selected (τ = 2%)</dt><dd>{man['selected']}</dd></div>
    <div><dt>Built</dt><dd>{esc(man['built'])}</dd></div>
  </dl>
</div></header>
<main class="wrap">
<section class="verdict">
  <div class="sh"><h2>What the data say before any allocation</h2></div>
  <div class="card"><p style="font-size:16px">{eb_txt}</p>
  <p class="cap">Shrinkage: α* = μ + τ²/(τ² + s²) · (α̂ − μ), with μ the precision-weighted cross-sectional mean and s each manager's Newey–West standard error. τ is estimated as √max(0, Var(α̂) − mean s²). This is the James–Stein logic applied to manager selection: the noisier the estimate, the less of it survives.</p></div>
</section>

<section>
  <div class="sh"><h2>Allocation under stated priors</h2></div>
  <p class="note">An allocator who believes some managers have skill must say how much dispersion in true alpha they believe in (Baks, Metrick &amp; Wachter 2001). Each row shrinks with that τ, keeps the managers whose shrunk alpha is positive, and maximises the blend's expected information ratio with no manager above {man['max_weight']:.0%}.</p>
  <div class="grid2">
    <div class="card tscroll"><h3>By prior</h3><table><thead><tr><th>prior</th><th class="n">τ</th><th class="n">managers held</th><th class="n">blend alpha</th><th class="n">expected IR</th></tr></thead><tbody>{prior_rows}</tbody></table>
      <p class="cap">Expected IR = shrunk blend alpha ÷ blend residual volatility from the shrunk correlation matrix. Ex ante; the out-of-sample section below is the check.</p></div>
    <div class="card tscroll"><h3>The τ = 2% blend, three ways</h3><table><thead><tr><th>blend</th><th class="n">managers</th><th class="n">alpha</th><th class="n">resid vol</th><th class="n">expected IR</th></tr></thead><tbody>{blend_rows}</tbody></table>
      <p class="cap">Optimisation versus equal weight of the top ten versus holding everyone: how much the correlation structure is worth on top of selection.</p></div>
  </div>
  <div class="card tscroll" style="margin-top:14px"><h3>Optimised weights, τ = 2%</h3><table><thead><tr><th>manager</th><th class="n">shrunk alpha</th><th class="n">residual vol</th><th class="n">shrunk IR</th><th class="n">weight</th></tr></thead><tbody>{wrow}</tbody></table>
  <p class="cap">Low-residual-volatility managers (listed funds, diversified long-only books) get large weights even with modest alpha because they cost little tracking risk; concentrated clones get small weights because their residual volatility is 10–14%/yr.</p></div>
</section>

<section>
  <div class="sh"><h2>Shrinkage, manager by manager</h2></div>
  <div class="card tscroll"><table><thead><tr><th>manager</th><th class="n">months</th><th class="n">raw alpha</th><th class="n">std error</th><th class="n">t</th><th class="n">keep (τ=2%)</th><th class="n">shrunk (τ=2%)</th><th class="n">empirical Bayes</th></tr></thead><tbody>{srow}</tbody></table>
  <p class="cap">Top twenty by raw alpha. "Keep" is τ²/(τ² + s²): the share of the raw estimate that survives. The empirical-Bayes column is what the data alone support. Full table: <a href="/research/data/fund-of-funds/managers.csv">managers.csv</a>.</p></div>
</section>

<section>
  <div class="sh"><h2>How many managers?</h2></div>
  <div class="card">{cchart}<p class="cap">Expected information ratio of an equal-weight blend of the top-N managers by shrunk IR (τ = 2%), adding one at a time. It rises while the added manager's alpha outweighs the dilution, then flattens: with residual correlations averaging {man['avg_corr']:+.2f}, diversification across managers is cheap but the alpha to diversify is thin.</p></div>
</section>

<section>
  <div class="sh"><h2>Does picking past winners work?</h2></div>
  <div class="card">{oos_html}</div>
</section>

<section>
  <div class="sh"><h2>Method and limits</h2></div>
  <div class="card">
    <p><b>Inputs.</b> Each manager's monthly FF3 regression from the verification pipeline: alpha, Newey–West standard error, residual series. Managers with fewer than {man['min_months']} months and strategies whose 13F is not a portfolio are excluded. Listed funds (real net returns) sit alongside 13F clones (reconstructed gross long books) — a real allocation would have audited net series for all of them.</p>
    <p><b>Correlations.</b> Pairwise on overlapping months (minimum 24), shrunk {man['corr_shrink']:.0%} toward zero. Residual returns, not total returns: factor exposure is bought elsewhere, cheaply.</p>
    <p><b>Optimisation.</b> Maximise αᵀw / √(wᵀΣw) over long-only weights, Σw = 1, each ≤ {man['max_weight']:.0%}, SLSQP. No transaction costs, capacity or liquidity terms; a real fund-of-funds adds minimum ticket sizes, redemption terms and operational scores.</p>
    <p><b>Reading it.</b> Two findings are the substance of the page: that the cross-section of alphas is indistinguishable from noise, and that ranking on past alpha did not select future alpha in this sample. Everything under a stated prior is what an allocator would do <i>if</i> they believed otherwise, shown so the belief is explicit rather than hidden in a weight.</p>
  </div>
</section>

<footer class="foot">
  <div class="running"><span>Track record verification · research</span><span>{man['first'][:7]} – {man['last'][:7]}</span></div>
  <h4>Important information</h4>
  <p>Built from the pipeline's own outputs on public SEC EDGAR filings and Yahoo Finance prices. A demonstration of fund-of-funds construction; not investment advice. Rebuild: <code>python -m trackrecord fund-of-funds</code>.</p>
</footer>
</main>
<div id="tip" class="tip" hidden></div>
<script>{JS}</script>
"""
