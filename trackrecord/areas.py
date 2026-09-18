"""The site organised the way the job is: two areas, one tool per responsibility.

    Active             signal research, risk model, portfolio construction, independent verification
    External managers  manager verification and screening, due-diligence memos, holdings
                       research, manager decay model, fund-of-funds construction

Every card's numbers are read from the committed CSVs under data/research/ (or the
leaderboard), so nothing on these pages can go stale, and a tool whose data is missing
is simply not shown rather than shown with a broken claim.
"""
from __future__ import annotations

import csv
import html
import json
from pathlib import Path

from .explain import CSS as EXPLAIN_CSS

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "data" / "research"

AREAS = {
    "active": ("Active portfolios", "/active",
               "One backtested mandate: pick a stock ranking rule (a signal), measure its risk, "
               "turn the ranks into trades, and check the arithmetic — told as objective, process, product."),
    "external": ("Manager Analysis", "/external",
                 "These tools evaluate outside managers: verify each record, separate skill from market exposure, "
                 "test it against luck, write the memo, and combine the managers that pass into a fund of funds."),
}


def _rows(path, key=None):
    f = R / path
    if not f.exists():
        return {}
    with f.open() as fh:
        rs = list(csv.DictReader(fh))
    return {r[key]: r for r in rs} if key else rs


def _man(path):
    f = R / path
    try:
        return json.loads(f.read_text()) if f.exists() else {}
    except Exception:
        return {}


def _num(x, d=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return d


def tool_cards() -> dict[str, list[dict]]:
    """Live tools by area.  Each: href, eyebrow, title, what, stats [(value, label)]."""
    from .explain import plain
    cards = {"active": [], "external": []}

    # ---- active
    lab, labman = _rows("alpha-lab/summary.csv", "signal"), _man("alpha-lab/manifest.json")
    if lab and labman:
        tested = {k: r for k, r in lab.items() if k not in ("linear", "xgboost")}
        best = max(tested.values(), key=lambda r: _num(r.get("spread_t"), -99))
        kept = [r for r in tested.values() if _num(r.get("spread_t"), 0) >= 2]
        hh = labman.get("head_to_head", {})
        cards["active"].append(dict(href="/research/alpha-lab", eyebrow="Which stock characteristics predict next month's return?",
            title="Signal Research",
            what=f"Eight stock characteristics, such as value, momentum and profitability, are tested for whether they predicted next month's return across about {labman.get('universe_avg', '')} stocks. "
                 f"One of them survives the test, and it is the signal the portfolio on the construction page trades.",
            stats=[(f"{len(kept)} of {len(tested)}", "signals with evidence behind them"),
                   (f"{_num(best.get('spread_ann'), 0) * 100:+.1f}%", f"best signal, {plain(best.get('label', best.get('signal', ''))).split(' (')[0].lower()}: top tenth minus bottom tenth a year"),
                   (f"{_num(hh.get('ic_diff_t'), 0):+.1f}", "learned model versus simple average")]))
    risk = _man("risk-model/manifest.json")
    if risk.get("factors"):
        cards["active"].append(dict(href="/research/risk-model", eyebrow="How much risk is a portfolio taking, and where does it come from?",
            title="Factor Risk Model",
            what="A risk model explains each stock's monthly return by its exposure to the market, to eight styles and to its industry. "
                 "It predicts how much any portfolio will move, splits that risk into what comes from those exposures and what is specific to the stocks held, and is tested against what actually happened.",
            stats=[(f"{risk.get('factors')}", "factors"), (f"{risk.get('industries')}", "industries"),
                   (f"{_num(risk.get('r2_avg'), 0):.0%}", "of monthly returns explained")]))
    con, conman = _rows("construction/summary.csv", "key"), _man("construction/manifest.json")
    if con and conman:
        P = con.get("portfolio", {})
        cards["active"].append(dict(href="/research/construction", eyebrow="How does a signal become a list of trades?",
            title="Portfolio Construction",
            what="An optimiser turns the signal into target weights that respect the mandate's limits on position size, sector tilts and turnover, and then into the list of trades a dealer would receive. "
                 "The page shows what those limits cost and what the portfolio delivered.",
            stats=[(f"{_num(P.get('active_return'), 0) * 100:+.1f}%", "active return per year"), (f"{_num(P.get('information_ratio'), 0):.2f}", "information ratio"),
                   (f"{conman.get('rebalances', '')}", "rebalances")]))
    rv = _man("r-verify/manifest.json")
    if rv.get("managers"):
        cards["active"].append(dict(href="/research/r-verify", eyebrow="Are the numbers right?",
            title="Independent Verification",
            what="Every headline number on the site was recalculated a second time by a separate implementation written from scratch, and the two sets of results were compared. "
                 "They agree to the limit of computer precision.",
            stats=[(f"{rv.get('managers')}", "managers"), (f"{rv.get('checks', 0):,}", "numbers checked"), (f"{rv.get('max_abs_diff', 0):.0e}", "largest difference")]))

    # ---- external
    sig, sigman = _rows("13f-signals/summary.csv", "key"), _man("13f-signals/manifest.json")
    if sig and sigman:
        t = _num(sig.get("BEST1", {}).get("carhart_t"))
        cards["external"].append(dict(href="/research/13f-signals", eyebrow="Can you make money by copying what the best managers own?",
            title="Holdings Research",
            what=f"Every manager's public quarterly holdings are turned into portfolios of their biggest positions, their most crowded stocks and their newest purchases, and tested for whether they beat the market. "
                 f"{'The finding is that nothing survives the 45-day delay before the holdings become public.' if t is None or abs(t) < 2 else 'The finding is that one spread survives the 45-day delay before the holdings become public.'}",
            stats=[(f"{sigman.get('managers', '')}", "managers"), (f"{sigman.get('quarters', '')}", "quarters"), (f"{_num(sigman.get('positions'), 0):,.0f}", "positions")]))
    dec, decman = _rows("decay/summary.csv", "model"), _man("decay/manifest.json")
    if dec and decman.get("managers"):
        xg, lg, ps = dec.get("xgboost", {}), dec.get("logistic", {}), dec.get("persist", {})
        best_t = max(_num(r.get("auc_cs_t"), 0) for r in dec.values())
        cards["external"].append(dict(href="/research/decay", eyebrow="Can a manager's filings say who will lag next year?",
            title="Manager Decay Model",
            what=f"For {decman['managers']} managers at every quarterly filing, the portfolio's concentration, turnover, crowding, recent trades and recent performance are used to predict who will trail the market over the next twelve months. "
                 f"{'The finding is that nothing detectable predicts it, and the page shows how a carelessly built test would claim otherwise.' if best_t < 2 else 'The finding is a weak signal, and the page shows how a carelessly built test would overstate it.'}",
            stats=[(f"{_num(xg.get('auc_cs_mean'), 0.5):.3f}", "learned model"), (f"{_num(lg.get('auc_cs_mean'), 0.5):.3f}", "regression"),
                   (f"{_num(ps.get('auc_cs_mean'), 0.5):.3f}", "last year's laggards")]))
    fof = _man("fund-of-funds/manifest.json")
    if fof.get("managers"):
        cards["external"].append(dict(href="/research/fund-of-funds", eyebrow="How should money be split across the managers that pass?",
            title="Fund of Funds",
            what="Each manager's measured skill is discounted for how noisy its estimate is, the managers' returns are checked for overlap, and the money is allocated to the combination with the best expected result. "
                 "The page shows how many managers diversification actually rewards.",
            stats=[(f"{fof.get('managers')}", "candidates"), (f"{fof.get('selected')}", "selected"), (f"{_num(fof.get('ir_blend'), 0):.2f}", "expected information ratio")]))
    return cards


def method_card() -> dict | None:
    """The due-diligence-on-itself page, shown once the checks behind it have been built."""
    m = _man("limits/manifest.json")
    if not m.get("managers"):
        return None
    return dict(href="/research/limits", eyebrow="What would a reviewer object to?",
                title="Due Diligence on This Work",
                what="Every tool here puts hard questions to an outside manager. This page puts the same questions to the work itself: "
                     "why this set of stocks, what is missing from it, whether the portfolio result matches the signal behind it, what disclosed "
                     "holdings can and cannot say about a manager, and how current the data is. Each answer is a number, and the unflattering ones are kept.",
                stats=[(f"{m.get('widest_universe', 0):,}", "widest set of stocks tested"),
                       (f"{_num(m.get('coverage_last'), 0):.0%}", "of holdings reach a price"),
                       (f"{m.get('long_short', 0)} of {m.get('managers', 0)}", "records that are a long book only")])


CARD_CSS = EXPLAIN_CSS + """<style>
.rcards{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(440px,100%),1fr));gap:14px;margin:6px 0 10px}
.rcard{display:flex;flex-direction:column;background:var(--surface);border:1px solid var(--line);border-top:2px solid var(--gold);padding:18px 20px;text-decoration:none;color:var(--ink)}
.rcard .eyeb{font:600 9px/1 var(--sans);letter-spacing:.22em;text-transform:uppercase;color:var(--muted)}
.rcard .rt{font:400 21px/1.2 var(--serif);color:var(--navy);margin:8px 0 6px}
.rcard .rw{font-size:13.5px;color:var(--ink2);line-height:1.5;flex:1}
.rcard .rs{display:flex;gap:18px;margin:14px 0 0;flex-wrap:wrap}.rcard .rs div{display:grid;gap:3px}
.rcard .rs .v{font:600 16px/1 var(--sans);color:var(--navy)}
.rcard .rs .l{font:600 8.5px/1.2 var(--sans);letter-spacing:.14em;text-transform:uppercase;color:var(--muted)}
.rcard{position:relative}.rcard .rt a{color:inherit;text-decoration:none}.rcard .rt a::after{content:"";position:absolute;inset:0}
.rcard .rs div{position:relative;z-index:1}.rcard .rs div:has(details[open]){flex-basis:100%}.rcard details.how{margin-top:6px}.rcard details.how .howb{font-size:12.5px}
.rcard .go{margin-top:14px;font:600 9.5px/1 var(--sans);letter-spacing:.18em;text-transform:uppercase;color:var(--gold)}
.rcard:hover{border-color:var(--gold)}
.areas{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(300px,100%),1fr));gap:14px;margin:26px 0 0}
.area{display:grid;gap:8px;padding:22px 24px;text-decoration:none;border:1px solid var(--goldl);color:var(--coverink);background:rgba(232,228,218,.04)}
.area .t{font:400 24px/1.15 var(--serif)}.area .s{font-size:13.5px;line-height:1.5;color:var(--covermuted)}.area .n{font:600 9.5px/1.3 var(--sans);letter-spacing:.18em;text-transform:uppercase;color:var(--goldl);margin-top:6px}
.area:hover{filter:brightness(1.12)}
.stats{display:flex;flex-wrap:wrap;gap:22px 40px;margin-top:30px;padding-top:18px;border-top:1px solid rgba(232,228,218,.18)}.stats div{display:grid;gap:4px}.stats dt{font:600 9px/1 var(--sans);letter-spacing:.22em;text-transform:uppercase;color:var(--covermuted)}.stats dd{margin:0;font:400 30px/1 var(--serif);color:var(--coverink)}
</style>"""


def cards_html(cards: list[dict]) -> str:
    from .explain import note_html
    out = ""
    for c in cards:
        st = "".join(f"<div><span class='v'>{html.escape(str(v))}</span><span class='l'>{html.escape(l)}</span>{note_html(l, c['href'])}</div>" for v, l in c["stats"])
        out += (f"<div class='rcard'><div class='eyeb'>{html.escape(c['eyebrow'])}</div><div class='rt'><a href=\"{c['href']}\">{html.escape(c['title'])}</a></div>"
                f"<div class='rw'>{c['what']}</div><div class='rs'>{st}</div></div>")
    return out


def home_html(page, n_managers: int, n_scorable: int) -> str:
    cards = tool_cards()
    n_tools = sum(len(v) for v in cards.values())
    areas = "".join(f"<a class='area' href='{href}'><span class='t'>{html.escape(name)}</span><span class='s'>{html.escape(blurb)}</span>"
                    f"<span class='n'>{len(cards[k])} tool{'s' if len(cards[k]) != 1 else ''} &rarr;</span></a>" for k, (name, href, blurb) in AREAS.items() if cards[k])
    sections = "".join(f"<h2 data-n='{html.escape(name)}' id='{k}'>{html.escape(name)}</h2><p class='note'>{html.escape(blurb)}</p><div class='rcards'>{cards_html(cards[k])}</div>"
                       for k, (name, href, blurb) in AREAS.items() if cards[k])
    method = method_card()
    if method:
        sections += ("<h2 data-n='Method' id='method'>Before you trust any of it</h2>"
                     "<p class='note'>The same scrutiny the rest of the site applies to other people's records, applied to this one.</p>"
                     f"<div class='rcards'>{cards_html([method])}</div>")
    return page("Manager due diligence and active equity — public data", CARD_CSS + f"""<div class="banner"><b>Public data.</b> Every tool runs on public information: managers' quarterly holdings filings, listed funds' prices and the standard academic market factors. Nothing here is investment advice.</div>
<header class="cover"><div class="cover-in"><div class="eyebrow">Quantitative portfolio management · global equity</div><div class="rule"></div><h1>Due diligence on outside managers —<br>and a stock portfolio on public data</h1>
<p class="sub">Reconstruct each manager from disclosed holdings, test which stock ranking scores predicted next month's return, turn the one that survives into a constrained trade list, and check every headline number a second time. Findings stay on the page even when they are nothing.</p>
<div class="areas">{areas}</div>
<dl class="stats"><div><dt>Tools</dt><dd>{n_tools}</dd></div><div><dt>Managers in the system</dt><dd>{n_managers}</dd></div><div><dt>Scorable today</dt><dd>{n_scorable}</dd></div></dl>
</div></header>
<main class="wrap">{sections}
<div class="foot">Built by Simon Weardon. Benchmark and factors: Kenneth R. French Data Library. Prices and sectors: Yahoo Finance. Holdings and company accounts: SEC EDGAR. Past performance is not indicative of future results, and nothing here is investment advice.</div>
</main>""", is_home=True)


def area_html(page, key: str, extra_body: str = "", extra_head: str = "") -> str:
    if key == "active" and not extra_body:
        from .story import active_html
        return active_html(page)
    name, href, blurb = AREAS[key]
    cards = tool_cards()[key]
    body = (f"<div class='rcards'>{cards_html(cards)}</div>" if cards else "<p class='note'>Tools for this area are being built.</p>")
    return page(name, CARD_CSS + extra_head + f"""<header class="cover"><div class="cover-in"><div class="eyebrow">Global equity · {html.escape(name.lower())}</div><div class="rule"></div><h1>{html.escape(name)}</h1>
<p class="sub">{html.escape(blurb)}</p></div></header>
<main class="wrap"><h2 data-n="Tools">What you can do here</h2>{body}{extra_body}</main>""", current=href)
