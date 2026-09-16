"""Pages for the Simulation tab: targets → suggested portfolio, side-by-side comparison, the
picker, and the results page (backtest with fees, scores, links to the dashboard and memo)."""
from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .dashboard import CSS as DASH_CSS, JS as DASH_JS, diverging_bars, esc, line_chart, pct, num
from .simulate import OUT, SIMS, SLEEVES, Targets, candidates, residual_correlation, stats_of

def _dash_rules() -> str:
    keep = (".chart", ".k ", ".k.", ".tip", ".tiles", ".tile", ".tl", ".tv", ".td", ".sv", ".of", ".legend", ".grid2", ".cap", ".muted", ".tscroll")
    return "\n".join(l for l in DASH_CSS.splitlines() if l.strip().startswith(keep))


SIM_CSS = """<style>:root{--accent:var(--navy);--accent-l:var(--gold);--accent-wash:rgba(27,42,65,.07);--dim:var(--muted);--dim-strong:var(--ink2);--s3:#5f7a5e;--pos:var(--navy);--neg:var(--crit);--hair:var(--line);--ink-2:var(--ink2);--surface-2:var(--page);--cover-ink:var(--coverink);--good-wash:#e3e9dd;--warn:var(--gold)}
@media(prefers-color-scheme:dark){:root{--accent:#d9c48f;--accent-l:#a89468;--accent-wash:rgba(217,196,143,.12);--dim:#8a9ab4;--s3:#7fa384;--pos:#e8e4da;--neg:#e07a6c}}</style><style>""" + _dash_rules() + """</style><style>
.sim-grid{display:grid;grid-template-columns:1fr;gap:18px}
.panel{background:var(--surface);border:1px solid var(--line);border-top:2px solid var(--gold);padding:18px 20px}
.panel h3{font:400 20px/1.2 var(--serif);color:var(--navy);margin:0 0 6px}.panel p.hint{color:var(--ink2);font-size:13px;margin:0 0 12px}
.sliders{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px 22px}
.sl{display:grid;gap:4px}.sl label{font:600 9px/1.2 var(--sans);letter-spacing:.16em;text-transform:uppercase;color:var(--muted);display:flex;justify-content:space-between}
.sl label output{font:600 12px var(--sans);color:var(--navy);letter-spacing:0;text-transform:none}
.sl input[type=range]{width:100%;accent-color:var(--gold)}
.styles{display:flex;flex-wrap:wrap;gap:8px 14px;margin:8px 0 0}.styles label{font-size:13px;display:flex;gap:6px;align-items:center}
.row{display:flex;flex-wrap:wrap;gap:12px;align-items:center;margin-top:12px}
.btn2{display:inline-block;font:600 10px/1 var(--sans);letter-spacing:.18em;text-transform:uppercase;padding:11px 18px;background:var(--navy);color:var(--coverink);border:0;cursor:pointer;text-decoration:none}
.btn2.gold{background:var(--goldl);color:#1b2a40}.btn2.ghost{background:transparent;color:var(--navy);border:1px solid var(--navy)}
@media(prefers-color-scheme:dark){.btn2{background:var(--goldl);color:#1b2a40}.btn2.ghost{color:var(--goldl);border-color:var(--goldl)}}
.pick{font-size:13.5px}.pick td{padding:7px 8px}.pick input[type=number]{width:64px;font:13px var(--sans);padding:4px 6px;border:1px solid var(--line);background:var(--page);color:var(--ink)}
.pick tr.sel td{background:var(--accent-wash,rgba(27,42,65,.07))}.pick .st{color:var(--ink2);font-size:12px}
.chk{display:flex;gap:8px;align-items:center;font-size:13.5px}
.checks td.ok{color:var(--good);font-weight:700}.checks td.miss{color:var(--crit);font-weight:700}
.tools2{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin:10px 0}.tools2 input[type=search]{font:14px var(--serif);padding:7px 10px;border:1px solid var(--line);background:var(--surface);color:var(--ink);flex:1;min-width:200px}
.fee{display:flex;flex-wrap:wrap;gap:14px;align-items:end}.fee label{display:grid;gap:4px;font:600 9px/1.2 var(--sans);letter-spacing:.16em;text-transform:uppercase;color:var(--muted)}
.fee input,.fee select{font:14px var(--sans);padding:6px 8px;border:1px solid var(--line);background:var(--page);color:var(--ink)}
</style>"""


def _f(v, fmt="{:.1%}", dash="—"):
    try:
        return dash if v is None or pd.isna(v) else fmt.format(v)
    except Exception:
        return dash


def simulate_html(page, targets: Targets, suggested: dict | None, selected: dict[str, float], compare: dict | None, fee: dict, error: str | None = None, alloc: dict | None = None) -> str:
    alloc = alloc or {"managers": 1.0}
    cands = candidates()
    styles = sorted({c["style"] for c in cands if c["style"]})
    by_key = {c["key"]: c for c in cands}
    # ---- targets panel
    def sl(name, label, lo, hi, step, val, fmt_js):
        return (f"<div class='sl'><label>{label} <output id='o_{name}'></output></label>"
                f"<input type='range' name='{name}' min='{lo}' max='{hi}' step='{step}' value='{val}' oninput=\"document.getElementById('o_{name}').value={fmt_js}\"></div>")
    sliders = "".join([
        sl("tr", "Target return /yr", 0, 0.30, 0.005, targets.target_return, "(this.value*100).toFixed(1)+'%'"),
        sl("mv", "Max volatility", 0.04, 0.35, 0.005, targets.max_vol, "(this.value*100).toFixed(1)+'%'"),
        sl("md", "Max drawdown", 0.05, 0.70, 0.01, targets.max_drawdown, "'−'+(this.value*100).toFixed(0)+'%'"),
        sl("sh", "Min Sharpe", 0, 2.0, 0.05, targets.min_sharpe, "Number(this.value).toFixed(2)"),
        sl("te", "Max tracking error vs market", 0.01, 0.20, 0.005, targets.max_te, "(this.value*100).toFixed(1)+'%'"),
        sl("n", "Managers", 2, 12, 1, targets.n_managers, "this.value"),
        sl("mw", "Max weight per manager", 0.10, 0.60, 0.05, targets.max_weight, "(this.value*100).toFixed(0)+'%'"),
        sl("mm", "Min track record (months)", 36, 180, 12, targets.min_months, "this.value"),
    ])
    style_boxes = "".join(f"<label><input type='checkbox' name='st' value='{esc(s)}' {'checked' if (not targets.styles or s in targets.styles) else ''}> {esc(s)}</label>" for s in styles)
    prefer = "".join(f"<option value='{v}' {'selected' if targets.prefer == v else ''}>{l}</option>" for v, l in [("wealth", "wealth-management score"), ("alpha_max", "alpha-maxing score"), ("t", "FF3 alpha t-statistic")])
    sug_html = ""
    if suggested is not None:
        if suggested.get("ok"):
            rows = "".join(f"<tr><td>{esc(a)}</td><td class='n'>{_f(b, '{:.2f}') if 'Sharpe' in a else _f(b)}</td><td class='n'>{_f(c, '{:.2f}') if 'Sharpe' in a else _f(c)}</td><td class='{'ok' if d else 'miss'}'>{'hit' if d else 'missed'}</td></tr>" for a, b, c, d in suggested["checks"])
            wrows = "".join(f"<tr><td><b>{esc(suggested['names'].get(k, k))}</b> <span class='st'>{esc(by_key.get(k, {}).get('style', ''))}</span></td><td class='n'>{v:.1%}</td></tr>" for k, v in sorted(suggested["weights"].items(), key=lambda kv: -kv[1]))
            sug_html = (f"<div class='panel'><h3>Suggested portfolio</h3><p class='hint'>{esc(suggested['how'])} — chosen from {suggested['pool']} candidates on {suggested['months']} common months ({suggested['first'][:7]} → {suggested['last'][:7]}). "
                        f"Historical, gross of fees: return {suggested['ann']:.1%}/yr vs market {suggested['mkt_ann']:.1%}, vol {suggested['vol']:.1%}, max drawdown {suggested['max_dd']:.0%}, Sharpe {suggested['sharpe']:.2f}, tracking error {suggested['te']:.1%}, IR {_f(suggested['ir'], '{:.2f}')}. "
                        f"It is pre-filled in the picker below — change anything, then Simulate.</p>"
                        f"<div class='grid2'><div class='tscroll'><table class='checks'><thead><tr><th>target</th><th class='n'>achieved</th><th class='n'>target</th><th></th></tr></thead><tbody>{rows}</tbody></table></div>"
                        f"<div class='tscroll'><table><thead><tr><th>manager</th><th class='n'>weight</th></tr></thead><tbody>{wrows}</tbody></table></div></div>"
                        f"<p class='cap'>A suggestion is a historical fit under your caps — the same past that a memo warns you not to extrapolate. Missed targets mean the history could not deliver them; the tool says so rather than bending the numbers.</p></div>")
        else:
            sug_html = f"<div class='panel'><h3>No suggestion</h3><p class='hint'>{esc(suggested.get('reason', ''))}</p></div>"
    # ---- compare block
    cmp_html = ""
    if compare:
        keys = compare["keys"]; S = compare["stats"]; C = compare["corr"]
        head = "<tr><th>manager</th><th class='n'>months</th><th class='n'>return /yr</th><th class='n'>vol</th><th class='n'>Sharpe</th><th class='n'>max DD</th><th class='n'>FF3 α</th><th class='n'>t</th><th class='n'>β</th><th class='n'>SMB</th><th class='n'>HML</th><th class='n'>down cap</th><th class='n'>TE</th><th class='n'>IR</th><th class='n'>alpha-max</th><th class='n'>wealth</th></tr>"
        rows = "".join(f"<tr><td><b>{esc(by_key.get(k, {}).get('name', k))}</b><br><span class='st'>{esc(by_key.get(k, {}).get('style', ''))}</span></td><td class='n'>{S[k].get('n', '')}</td><td class='n'>{_f(S[k].get('ret'))}</td><td class='n'>{_f(S[k].get('vol'))}</td><td class='n'>{_f(S[k].get('sharpe'), '{:.2f}')}</td><td class='n'>{_f(S[k].get('max_dd'), '{:.0%}')}</td><td class='n'>{_f(S[k].get('alpha'), '{:+.1%}')}</td><td class='n'>{_f(S[k].get('t'), '{:+.1f}')}</td><td class='n'>{_f(S[k].get('beta'), '{:.2f}')}</td><td class='n'>{_f(S[k].get('smb'), '{:+.2f}')}</td><td class='n'>{_f(S[k].get('hml'), '{:+.2f}')}</td><td class='n'>{_f(S[k].get('down_cap'), '{:.2f}')}</td><td class='n'>{_f(S[k].get('te'))}</td><td class='n'>{_f(S[k].get('ir'), '{:.2f}')}</td><td class='n'>{_f(S[k].get('alpha_max'), '{:.0f}')}</td><td class='n'>{_f(S[k].get('wealth'), '{:.0f}')}</td></tr>" for k in keys)
        ch = "".join(f"<th class='n'>{esc(by_key.get(k, {}).get('name', k)[:14])}</th>" for k in C.columns)
        cr = "".join("<tr><td>" + esc(by_key.get(r, {}).get('name', r)[:22]) + "</td>" + "".join(f"<td class='n'>{_f(C.loc[r, c], '{:+.2f}')}</td>" for c in C.columns) + "</tr>" for r in C.index)
        cmp_html = (f"<div class='panel'><h3>Side by side</h3><p class='hint'>Each manager's own verified numbers, from its dashboard. Residual correlation is what diversification acts on: after the market and factors, do these managers win and lose together?</p>"
                    f"<div class='tscroll'><table>{head}<tbody>{rows}</tbody></table></div>"
                    f"<h4 style='margin:16px 0 6px'>Residual correlation (FF3), overlapping months</h4><div class='tscroll'><table><thead><tr><th></th>{ch}</tr></thead><tbody>{cr}</tbody></table></div>"
                    f"{compare.get('chart', '')}</div>")
    # ---- picker
    ordered = sorted(cands, key=lambda c: (0 if c["key"] in selected else 1, -(c.get("ff3_t") or -99)))
    prow = ""
    for c in ordered:
        k = c["key"]; sel = k in selected; w = selected.get(k, 0.0)
        prow += (f"<tr class='{'sel' if sel else ''}' data-s='{esc((c['name'] + ' ' + c['style'] + ' ' + c.get('manager', '')).lower())}'><td><input type='checkbox' name='sel' value='{esc(k)}' {'checked' if sel else ''} onchange='tog(this)'></td>"
                 f"<td><b>{esc(c['name'])}</b>{' <span class=st>' + esc(c['manager']) + '</span>' if c.get('manager') else ''}<br><span class='st'>{esc(c['style'])} · {c['months']} mo{' · not built yet' if not c['built'] else ''}</span></td>"
                 f"<td class='n'>{_f(c.get('ff3_t'), '{:+.1f}')}</td><td class='n'><input type='number' name='w_{esc(k)}' min='0' max='100' step='1' value='{round(w * 100) if sel else ''}' {'' if sel else 'disabled'}></td>"
                 f"<td class='n'><a href='/{'t' if c['kind'] == 'ticker' else 'f'}/{esc(k)}/{'memo' if c['kind'] != 'ticker' else 'memo'}' class='st'>memo</a></td></tr>")
    def asl(key, label, val):
        return (f"<div class='sl'><label>{label} <output id='oa_{key}'></output></label>"
                f"<input type='range' name='a_{key}' min='0' max='100' step='5' value='{round(val * 100)}' oninput=\"document.getElementById('oa_{key}').value=this.value+'%';asum()\"></div>")
    alloc_html = ("<h4 style='margin:16px 0 4px'>Asset allocation</h4><p class='hint'>What share of the total portfolio goes to the fund of managers above, and what to the other asset classes (public total-return ETFs; cash = T-bills). Shares are normalised to 100%.</p>"
                  "<div class='sliders'>" + "".join(asl(k, lab, alloc.get(k, 0.0)) for k, (lab, _) in SLEEVES.items()) + "</div><p class='st' id='asum'></p>")
    fee_html = (f"<div class='fee'><label>Management fee /yr<input type='number' name='mgmt' min='0' max='5' step='0.05' value='{fee['mgmt'] * 100:g}'>%</label>"
                f"<label>Performance fee<input type='number' name='perf' min='0' max='50' step='1' value='{fee['perf'] * 100:g}'>% of gains above HWM</label>"
                f"<label>Rebalancing<select name='reb'><option value='monthly' {'selected' if fee['reb'] == 'monthly' else ''}>monthly, back to target</option><option value='drift' {'selected' if fee['reb'] == 'drift' else ''}>buy and hold (drift)</option></select></label>"
                f"<label>Name<input type='text' name='name' placeholder='My portfolio' value='{esc(fee.get('name', ''))}' style='min-width:180px'></label></div>")
    js = """<script>
function tog(cb){const tr=cb.closest('tr'),inp=tr.querySelector('input[type=number]');inp.disabled=!cb.checked;tr.classList.toggle('sel',cb.checked);if(cb.checked&&!inp.value){inp.value='';}tot();}
function tot(){let s=0,n=0;document.querySelectorAll('.pick input[type=number]:not(:disabled)').forEach(i=>{s+=parseFloat(i.value)||0;n++;});document.getElementById('tot').textContent=n+' selected · weights sum '+s.toFixed(0)+'%'+(n&&Math.abs(s-100)>0.5?' (will be normalised to 100%)':'');}
function eq(){const inps=[...document.querySelectorAll('.pick input[type=number]:not(:disabled)')];inps.forEach(i=>i.value=(100/inps.length).toFixed(1));tot();}
function cmp(){const k=[...document.querySelectorAll('.pick input[type=checkbox]:checked')].map(c=>c.value);if(!k.length){alert('Pick at least one manager');return false;}location.href='/simulate?cmp='+k.join(',')+'#compare';return false;}
document.getElementById('q').addEventListener('input',e=>{const s=e.target.value.toLowerCase();document.querySelectorAll('.pick tbody tr').forEach(r=>{r.hidden=!(r.classList.contains('sel')||!s||r.dataset.s.includes(s));});});
function asum(){let s=0;document.querySelectorAll("input[name^='a_']").forEach(i=>s+=parseFloat(i.value)||0);document.getElementById('asum').textContent='allocation sum '+s.toFixed(0)+'%'+(Math.abs(s-100)>0.5?' (will be normalised to 100%)':'');}
document.querySelectorAll('.sl input[type=range]').forEach(r=>r.dispatchEvent(new Event('input')));tot();asum();
</script>"""
    err = f"<div class='banner' style='background:var(--crit);color:#fff'><b>Could not run</b> {esc(error)}</div>" if error else ""
    return page("Simulation", SIM_CSS + err + f"""<header class="cover"><div class="cover-in"><div class="eyebrow">Simulation · fund of managers</div><div class="rule"></div><h1>Simulation</h1>
<p class="sub">Say what you are looking for and get a suggested portfolio of managers; compare managers side by side; then blend your own pick, with fees and a rebalancing rule, and run it through the same verification, scores and memo as any single manager.</p></div></header>
<main class="wrap"><div class="sim-grid">
<form method="get" action="/simulate" class="panel" id="targets"><input type="hidden" name="suggest" value="1"><h3>What are you looking for?</h3>
<p class="hint">Set the numbers you want the portfolio to hit. The optimiser fits them on the managers' common history under your caps and reports which targets the history could deliver.</p>
<div class="sliders">{sliders}</div>
<div class="row"><span style="font:600 9px/1.2 var(--sans);letter-spacing:.16em;text-transform:uppercase;color:var(--muted)">Rank candidates by</span><select name="prefer" style="font:14px var(--sans);padding:6px 8px;border:1px solid var(--line);background:var(--page);color:var(--ink)">{prefer}</select></div>
<div class="styles">{style_boxes}</div>
<div class="row"><button class="btn2 gold" type="submit">Suggest a portfolio</button></div></form>
{sug_html}
<div id="compare">{cmp_html}</div>
<form method="post" action="/simulate/run" class="panel" id="picker"><h3>Pick managers and weights</h3>
<p class="hint">Tick managers and give weights (they are normalised to 100%). Compare shows them side by side without running anything; Simulate builds the blended portfolio and runs the full pipeline (about half a minute).</p>
<div class="tools2"><input type="search" id="q" placeholder="Filter managers by name, style or person"><span id="tot" class="st"></span><button type="button" class="btn2 ghost" onclick="eq()">Equal weights</button></div>
<div class="tscroll"><table class="pick"><thead><tr><th></th><th>manager</th><th class="n">FF3 t</th><th class="n">weight %</th><th></th></tr></thead><tbody>{prow}</tbody></table></div>
{alloc_html}
<h4 style="margin:16px 0 6px">Fees and rebalancing</h4>{fee_html}
<div class="row"><button type="button" class="btn2 ghost" onclick="return cmp()">Compare side by side</button><button class="btn2 gold" type="submit">Simulate this portfolio</button></div>
<p class="cap">Fees are charged to each manager on its own gains above its own high-water mark, then the sleeves are blended. A simulation is a what-if on public reconstructions; it is not a record and not investment advice.</p></form>
</div></main><div id="tip" class="tip" hidden></div>{js}<script>{DASH_JS}</script>""", current="/simulate")


def compare_block(keys: list[str]) -> dict:
    from .simulate import gross_series
    S = {k: stats_of(k) for k in keys}
    keys = [k for k in keys if S[k].get("n")]
    C = residual_correlation(keys)
    series = {k: gross_series(k) for k in keys}; series = {k: s for k, s in series.items() if s is not None}
    chart = ""
    if len(series) >= 1:
        idx = None
        for s in series.values():
            idx = s.index if idx is None else idx.intersection(s.index)
        idx = idx.sort_values()
        if len(idx) >= 12:
            cells = [d.strftime("%Y-%m") for d in idx]
            cls = ["s1", "s4", "s2", "s5", "s0", "s1l"]
            ser = [dict(name=k, values=[float(v) for v in (1 + series[k].reindex(idx)).cumprod()], cls=cls[i % len(cls)]) for i, k in enumerate(series)]
            chart = ("<h4 style='margin:16px 0 6px'>Growth of $1 on the common window</h4><div class='legend'>" + "".join(f"<span><span class='k {s['cls']}'></span>{esc(s['name'])}</span>" for s in ser) + "</div>"
                     + line_chart(cells, ser, height=260, width=860, y_fmt=lambda v: f"{v:.1f}×", y_log=True, end_labels=False, uid="cmpg"))
    return dict(keys=keys, stats=S, corr=C, chart=chart)


def sim_summary_html(page, sid: str, out_dir: Path, data_dir: Path) -> str | None:
    meta = json.loads((data_dir / "meta.json").read_text())
    if not (out_dir / "phase4" / "scores.csv").exists():
        return None
    bm = pd.read_csv(data_dir / "blend_monthly.csv"); bm["month"] = pd.PeriodIndex(bm.month, freq="M").to_timestamp("M"); bm = bm.set_index("month")
    from .reference import load_all
    ref = load_all(); ref.index = pd.DatetimeIndex(ref.index).to_period("M").to_timestamp("M")
    mkt = ref["US_MKT"].reindex(bm.index)
    sc = pd.read_csv(out_dir / "phase4" / "scores.csv")
    am = float(sc[sc.score == "Alpha-maxing score"].value.iloc[0]); wm = float(sc[(sc.score == "Wealth-management score") & (sc.component == "TOTAL")].value.iloc[0])
    comps = sc[(sc.score == "Wealth-management score") & (sc.component != "TOTAL") & (~sc.component.str.startswith("  "))]
    m = pd.read_csv(out_dir / "phase4" / "metrics.csv").set_index("metric")
    rr = pd.read_csv(out_dir / "phase4" / "regressions.csv"); ff3 = rr[(rr.factor_set == "US") & (rr.model == "FF3")].iloc[0]
    def mv(k, col="portfolio"):
        try: return float(m.loc[k, col])
        except Exception: return np.nan
    ann = lambda s: float((1 + s).prod() ** (12 / len(s)) - 1)
    g, n_, mk = bm.gross, bm.net, mkt.fillna(0)
    cells = [d.strftime("%Y-%m") for d in bm.index]
    chart = line_chart(cells, [dict(name="Market", values=[float(v) for v in (1 + mk).cumprod()], cls="s0"), dict(name="Portfolio, gross", values=[float(v) for v in (1 + g).cumprod()], cls="s1", emph=True),
                               dict(name="Portfolio, net of manager fees", values=[float(v) for v in (1 + n_).cumprod()], cls="s4")], height=280, width=860, y_fmt=lambda v: f"{v:.1f}×", y_log=True, end_labels=False, uid="simg")
    legend = "".join(f"<span><span class='k {c}'></span>{l}</span>" for l, c in [("Market", "s0"), ("Portfolio, gross", "s1"), ("Net of manager fees", "s4")])
    yrs = pd.DataFrame({"gross": g, "net": n_, "market": mk}).groupby(bm.index.year).apply(lambda x: (1 + x).prod() - 1)
    yrow = "".join(f"<tr><td>{y}</td><td class='n'>{r.gross:+.1%}</td><td class='n'>{r.net:+.1%}</td><td class='n'>{r.market:+.1%}</td><td class='n'>{(r.gross - r.market) * 1e4:+.0f}</td></tr>" for y, r in yrs.iterrows())
    ybars = diverging_bars([str(y) for y in yrs.index], [float(v) for v in (yrs.gross - yrs.market)], height=200, width=860, tips=[f"{y}: gross {r.gross:+.1%} vs market {r.market:+.1%}" for y, r in yrs.iterrows()])
    cum = (1 + g).cumprod(); dd = cum / cum.cummax() - 1; cumn = (1 + n_).cumprod(); ddn = cumn / cumn.cummax() - 1
    per = pd.DataFrame(meta["per_manager"]) if meta.get("per_manager") else pd.DataFrame(columns=["key", "weight", "gross", "net", "fee_drag", "vol"])
    cands = {c["key"]: c for c in candidates()}
    frow = "".join(f"<tr><td><b>{esc(cands.get(r.key, {}).get('name', r.key))}</b> <span class='muted'>{esc(cands.get(r.key, {}).get('style', ''))}</span></td><td class='n'>{r.weight:.0%}</td><td class='n'>{r.gross:.1%}</td><td class='n'>{r.net:.1%}</td><td class='n'>{r.fee_drag * 1e4:.0f} bps</td><td class='n'>{r.vol:.1%}</td></tr>" for _, r in per.sort_values("weight", ascending=False).iterrows())
    crow = "".join(f"<tr><td>{esc(r.component.strip())}</td><td class='n'>{float(r.value):.0f}</td><td class='n'>{float(r.weight) if pd.notna(r.weight) else ''}</td></tr>" for _, r in comps.iterrows())
    spec = meta["spec"]; alloc = spec.get("alloc", {"managers": 1.0})
    edit = ("/simulate?m=" + ",".join(f"{k}:{w:.4f}" for k, w in spec["managers"]) + f"&mgmt={spec['mgmt']}&perf={spec['perf']}&reb={spec['rebalance']}"
            + "&a=" + ",".join(f"{k}:{v:.4f}" for k, v in alloc.items()))
    sleeves = meta.get("sleeves") or []
    srow = "".join(f"<tr><td><b>{esc(x['label'])}</b></td><td class='n'>{x['weight']:.0%}</td><td class='n'>{x['ret']:.1%}</td><td class='n'>{x['vol']:.1%}</td><td class='n'>{x['contribution'] * 100:+.1f} pp</td></tr>" for x in sleeves)
    alloc_panel = (f"<div class='panel'><h3>Asset allocation</h3><p class='hint'>Each sleeve on the common window, gross of manager fees; contribution = weight × sleeve return (arithmetic, annualized).</p>"
                   f"<div class='tscroll'><table><thead><tr><th>sleeve</th><th class='n'>weight</th><th class='n'>return /yr</th><th class='n'>vol</th><th class='n'>contribution</th></tr></thead><tbody>{srow}</tbody></table></div>"
                   + (f"<p class='cap'>The fund-of-managers sleeve alone: {meta['mgr_ann_gross']:.1%}/yr gross, {meta['mgr_ann_net']:.1%}/yr net of manager fees.</p>" if meta.get('mgr_ann_gross') is not None else "")
                   + "</div>") if len(sleeves) > 1 else ""
    return page(f"{meta['name']} — Simulation", SIM_CSS + f"""<header class="cover"><div class="cover-in"><div class="eyebrow">Simulation · results</div><div class="rule"></div><h1>{esc(meta['name'])}</h1>
<p class="sub">{esc(meta['manager'])}. {esc(meta['style_note'])}. {meta['months']} months, {meta['first'][:7]} → {meta['last'][:7]}. Run through the same pipeline as every manager: the scores and the memo below are the standard ones.</p>
<div class="row" style="margin-top:18px"><a class="btn2 gold" href="/sims/{sid}/dashboard.html">Full dashboard</a><a class="btn2 gold" href="/sim/{sid}/memo">Due diligence memo</a><a class="btn2 ghost" style="color:var(--coverink);border-color:var(--goldl)" href="{edit}">Edit this portfolio</a></div></div></header>
<main class="wrap"><div class="sim-grid">
<div class="panel"><h3>Scores</h3><div class="tiles">
  <div class="tile score"><div class="tl">Alpha-maxing score</div><div class="tv"><span class="sv {'good' if am >= 70 else ''}">{am:.0f}</span><span class="of">/ 100</span></div><div class="td muted">50 + 10 × excess over the market (%/yr): {pct(mv('annualized return') - mv('annualized return', 'benchmark'), 1)}</div></div>
  <div class="tile score"><div class="tl">Wealth-management score</div><div class="tv"><span class="sv {'good' if wm >= 70 else ''}">{wm:.0f}</span><span class="of">/ 100</span></div><div class="td muted">skill 30% · risk-adjusted 25% · downside 25% · consistency 20%</div></div>
  <div class="tile"><div class="tl">FF3 alpha</div><div class="tv">{pct(ff3.alpha_annual, 1)}</div><div class="td muted">t = {ff3.t:+.2f} · β {ff3.b_MKT_RF:.2f} · SMB {ff3.b_SMB:+.2f} · HML {ff3.b_HML:+.2f}</div></div>
  <div class="tile"><div class="tl">Sharpe · IR</div><div class="tv">{num(mv('Sharpe (excess over RF)'))} · {num(mv('information ratio'))}</div><div class="td muted">tracking error {pct(mv('tracking error (ann.)'), 1, False)} · max DD {pct(mv('max drawdown (cell-level; understated on annual cells)'), 0, False)}</div></div>
</div><div class="tscroll" style="margin-top:10px"><table><thead><tr><th>wealth-management component</th><th class="n">score</th><th class="n">weight</th></tr></thead><tbody>{crow}</tbody></table></div>
<p class="cap">Scores are computed on the gross blend by the same fixed maps used for every manager, so this portfolio sits on the same scale as any row in Manager Analysis.</p></div>
{alloc_panel}
<div class="panel"><h3>Backtest</h3><p class="hint">The blended portfolio's history with {esc(meta['fee'])}, {esc(spec['rebalance'])} rebalancing, against the US market.</p>
<div class="legend">{legend}</div>{chart}
<div class="tiles" style="margin-top:12px">
  <div class="tile"><div class="tl">Gross /yr</div><div class="tv">{ann(g):.1%}</div><div class="td muted">market {ann(mk):.1%}</div></div>
  <div class="tile"><div class="tl">Net of manager fees /yr</div><div class="tv">{ann(n_):.1%}</div><div class="td muted">fee drag {(ann(g) - ann(n_)) * 1e4:.0f} bps/yr</div></div>
  <div class="tile"><div class="tl">Max drawdown</div><div class="tv">{dd.min():.0%}</div><div class="td muted">net {ddn.min():.0%} · market {((1 + mk).cumprod() / (1 + mk).cumprod().cummax() - 1).min():.0%}</div></div>
  <div class="tile"><div class="tl">Best / worst month</div><div class="tv small">{g.max():+.1%} / {g.min():+.1%}</div><div class="td muted">gross</div></div>
</div>
<h4 style="margin:16px 0 6px">Year by year</h4><div class="tscroll"><table><thead><tr><th>year</th><th class="n">gross</th><th class="n">net</th><th class="n">market</th><th class="n">excess bps</th></tr></thead><tbody>{yrow}</tbody></table></div>
<h4 style="margin:16px 0 6px">Excess over the market by year (gross)</h4>{ybars}</div>
{'<div class="panel"><h3>Fee breakdown</h3>' if per is not None and len(per) else '<div class="panel" hidden><h3>Fee breakdown</h3>'}<p class="hint">{esc(meta['fee'])}, charged to each manager on its own gains above its own high-water mark. The dashboard's "model-net" tile applies the same schedule once, at the portfolio level, which is slightly gentler.</p>
<div class="tscroll"><table><thead><tr><th>manager</th><th class="n">weight</th><th class="n">gross /yr</th><th class="n">net /yr</th><th class="n">fee drag</th><th class="n">vol</th></tr></thead><tbody>{frow}</tbody></table></div>
<p class="cap">Blend: gross {meta['ann_gross']:.1%}/yr, net {meta['ann_net']:.1%}/yr, drag {meta['fee_drag'] * 1e4:.0f} bps/yr. Performance fees compound the drag in good years and vanish in drawdowns, which is why net-of-fee volatility is lower than gross.</p></div>
</div></main><div id="tip" class="tip" hidden></div><script>{DASH_JS}</script>""", current="/simulate")
