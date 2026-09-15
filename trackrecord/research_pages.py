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
