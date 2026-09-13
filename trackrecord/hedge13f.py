"""13F long-only clone of a manager's disclosed US equity book.

What it is: each quarter, buy the manager's disclosed holdings at disclosed
weights once the 13F is public (filed ≤45 days after quarter end), hold with
drift until the next filing is public.  Monthly returns from adjusted closes.

What it is not: the fund's return.  No shorts, options, cash, leverage,
non-US holdings or intra-quarter trading, and every position is entered
~45 days late.  For concentrated long-biased managers it tracks the long book;
for multi-strategy / quant / macro shops it is not meaningful (flagged by style).

Data: SEC EDGAR (structured XML information tables exist from 2013 Q2);
CUSIP→ticker via SEC company_tickers.json name match, then OpenFIGI;
prices via yfinance (survivorship: delisted names drop out of the clone —
disclosed as `coverage`).  All raw pulls are cached under data/reference/edgar/.
"""
from __future__ import annotations

import gzip
import json
import re
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EDGAR = ROOT / "data" / "reference" / "edgar"
UA = None                       # set via set_contact(); SEC requires "Name email"
XML_ERA = "2013-06-30"          # first period with structured information tables
TOP_N = 60                      # positions kept per filing (by value)
MIN_COVERAGE = 0.6              # below this share of priced value, the month is NaN
MIN_POSITIONS = 5               # fewer priced names than this and the month is NaN (a clone, not a bet)
STALE_MONTHS = 4                # a filing's holdings are held at most this long without a newer filing

_last = [0.0]


def set_contact(contact: str) -> None:
    global UA
    UA = {"User-Agent": contact, "Accept-Encoding": "gzip, deflate"}


def _get(url: str, retries: int = 3) -> bytes:
    assert UA, "call set_contact('Name email@domain') first — SEC requires it"
    for k in range(retries):
        wait = 0.11 - (time.time() - _last[0])            # ≤ ~9 req/s
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.time()
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=40) as r:
                raw = r.read()
                return gzip.decompress(raw) if r.headers.get("Content-Encoding") == "gzip" else raw
        except urllib.error.HTTPError as e:
            if e.code in (403, 429, 503) and k < retries - 1:
                time.sleep(5 * (k + 1)); continue
            raise
        except Exception:
            if k < retries - 1:
                time.sleep(2 * (k + 1)); continue
            raise


# ---------------------------------------------------------------- filings

def list_13f(cik: str) -> list[dict]:
    """All 13F-HR filings (original, not amendments) for a CIK, incl. older pages."""
    cache = EDGAR / "submissions" / f"{int(cik):010d}.json"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists() and time.time() - cache.stat().st_mtime < 7 * 86400:
        return json.loads(cache.read_text())
    d = json.loads(_get(f"https://data.sec.gov/submissions/CIK{int(cik):010d}.json"))
    pages = [d["filings"]["recent"]]
    for f in d["filings"].get("files", []):
        pages.append(json.loads(_get(f"https://data.sec.gov/submissions/{f['name']}")))
    out = []
    for pg in pages:
        for i, form in enumerate(pg["form"]):
            if form == "13F-HR":
                out.append(dict(cik=str(int(cik)), accession=pg["accessionNumber"][i], filed=pg["filingDate"][i],
                                period=pg["reportDate"][i]))
    out.sort(key=lambda x: x["period"])
    cache.write_text(json.dumps(out))
    return out


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def fetch_holdings(cik: str, accession: str) -> list[dict] | None:
    """Parsed information table: [{cusip, name, value, shares, putcall}] or None if no XML table."""
    acc = accession.replace("-", "")
    cache = EDGAR / "13f" / str(int(cik)) / f"{acc}.json"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        return json.loads(cache.read_text())
    base = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc}/"
    try:
        idx = json.loads(_get(base + "index.json"))
    except Exception:
        cache.write_text("null"); return None
    names = [it["name"] for it in idx["directory"]["item"] if it["name"].lower().endswith(".xml")]
    names = [n for n in names if "primary_doc" not in n.lower()] + [n for n in names if "primary_doc" in n.lower()]
    rows = None
    for n in names:
        try:
            xml = _get(base + n)
        except Exception:
            continue
        if b"infoTable" not in xml and b"INFOTABLE" not in xml.upper():
            continue
        rows = []
        try:
            root = ET.fromstring(xml)
        except ET.ParseError:
            continue
        for it in root.iter():
            if _local(it.tag) != "infotable":
                continue
            rec = dict(cusip="", name="", value=0.0, shares=0.0, putcall="", cls="")
            for ch in it.iter():
                t = _local(ch.tag); v = (ch.text or "").strip()
                if t == "cusip": rec["cusip"] = v.upper()
                elif t == "nameofissuer": rec["name"] = v
                elif t == "titleofclass": rec["cls"] = v
                elif t == "value":
                    try: rec["value"] = float(v.replace(",", ""))
                    except ValueError: pass
                elif t == "sshprnamt":
                    try: rec["shares"] = float(v.replace(",", ""))
                    except ValueError: pass
                elif t == "putcall": rec["putcall"] = v.lower()
            if rec["cusip"]:
                rows.append(rec)
        if rows:
            break
    cache.write_text(json.dumps(rows))
    return rows


def fund_holdings(fund: dict, log=print) -> pd.DataFrame:
    """One row per (period, cusip): value summed across the manager's filer entities,
    puts/calls excluded, plus the filing date used for the rebalance."""
    frames = []
    for filer in fund["filers"]:
        fl = [f for f in list_13f(filer["cik"]) if f["period"] >= XML_ERA]
        for f in fl:
            rows = fetch_holdings(f["cik"], f["accession"])
            if not rows:
                continue
            df = pd.DataFrame(rows)
            df = df[df.putcall == ""]
            df = df[df.value > 0]
            if df.empty:
                continue
            df["period"] = f["period"]; df["filed"] = f["filed"]; df["cik"] = f["cik"]
            frames.append(df[["period", "filed", "cik", "cusip", "name", "cls", "value"]])
    if not frames:
        return pd.DataFrame(columns=["period", "filed", "cusip", "name", "cls", "value", "weight"])
    h = pd.concat(frames, ignore_index=True)
    # merge entities: same period -> sum by cusip; filing date = latest of the entities' filings
    filed = h.groupby("period").filed.max()
    g = h.groupby(["period", "cusip"]).agg(value=("value", "sum"), name=("name", "first"), cls=("cls", "first")).reset_index()
    g["filed"] = g.period.map(filed)
    g["weight_all"] = g.value / g.groupby("period").value.transform("sum")
    g = g.sort_values(["period", "value"], ascending=[True, False])
    g["rank"] = g.groupby("period").cumcount() + 1
    g = g[g["rank"] <= TOP_N].copy()
    g["weight"] = g.value / g.groupby("period").value.transform("sum")
    return g.reset_index(drop=True)


# ---------------------------------------------------------------- cusip -> ticker

_SUFFIX = {"INC", "CORP", "CORPORATION", "CO", "COMPANY", "LTD", "LIMITED", "PLC", "LLC", "LP", "L P", "HOLDINGS", "HLDGS", "HLDG",
           "GROUP", "GRP", "TRUST", "TR", "THE", "COM", "NEW", "ADR", "ADS", "ORD", "SHS", "DEL", "NV", "N V", "SA", "S A", "AG", "SE",
           "CL A", "CL B", "CL C", "CLASS A", "CLASS B", "CLASS C", "COMMON", "STOCK", "SPONSORED", "INTL", "INTERNATIONAL", "INC/DE"}


_PHRASES = ["CL A", "CL B", "CL C", "CLASS A", "CLASS B", "CLASS C", "N V", "S A", "L P", "SER A", "SER B", "NEW COM", "COM NEW", "SPONSORED ADR"]


def _norm(name: str) -> str:
    s = " " + re.sub(r"\s+", " ", re.sub(r"[^A-Z0-9 ]+", " ", name.upper())).strip() + " "
    for ph in _PHRASES:
        s = s.replace(f" {ph} ", " ")
    toks = [t for t in s.split() if t not in _SUFFIX]
    return " ".join(toks)


def sec_ticker_table() -> pd.DataFrame:
    cache = EDGAR / "company_tickers.json"
    if not cache.exists() or time.time() - cache.stat().st_mtime > 7 * 86400:
        cache.write_bytes(_get("https://www.sec.gov/files/company_tickers.json"))
    d = json.loads(cache.read_text())
    df = pd.DataFrame(list(d.values()))
    df["norm"] = df.title.map(_norm)
    df["ticker"] = df.ticker.str.replace(".", "-", regex=False)
    # prefer the shortest ticker per name (the primary class)
    df = df.sort_values("ticker", key=lambda s: s.str.len())
    return df.drop_duplicates("norm")


def map_cusips(holdings: pd.DataFrame, log=print, figi: bool = True) -> dict[str, str | None]:
    """cusip -> Yahoo ticker (or None).  Cached; OpenFIGI used only for names the SEC table can't match."""
    cache = EDGAR / "cusip_map.json"
    m: dict = json.loads(cache.read_text()) if cache.exists() else {}
    need = holdings.drop_duplicates("cusip")
    need = need[~need.cusip.isin(m) | need.cusip.map(lambda c: m.get(c) is None)]   # retry unmapped with the SEC table
    if len(need):
        sec = sec_ticker_table(); by_norm = dict(zip(sec.norm, sec.ticker))
        hit = 0
        for r in need.itertuples(index=False):
            key = _norm(r.name)
            t = by_norm.get(key)
            if t:
                m[r.cusip] = t; hit += 1
        log(f"  cusip map: {hit} matched by SEC name table, {len(need) - hit} left for OpenFIGI")
        cache.write_text(json.dumps(m))
    if figi:
        rest = [c for c in holdings.cusip.unique() if c not in m]
        if rest:
            _openfigi(rest, m, log)
            cache.write_text(json.dumps(m))
    return m


def _openfigi(cusips: list[str], m: dict, log=print) -> None:
    """Public OpenFIGI: 10 jobs / request, 25 requests / minute without a key."""
    url = "https://api.openfigi.com/v3/mapping"
    batch = 10; i = 0; fails = 0
    while i < len(cusips):
        chunk = cusips[i:i + batch]
        jobs = [{"idType": "ID_CUSIP", "idValue": c, "exchCode": "US"} for c in chunk]
        req = urllib.request.Request(url, data=json.dumps(jobs).encode(), headers={"Content-Type": "application/json", "User-Agent": "track-record-verification"})
        try:
            with urllib.request.urlopen(req, timeout=40) as r:
                res = json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code == 429 and fails < 20:
                fails += 1; time.sleep(35); continue
            log(f"  openfigi http {e.code}; stopping mapping"); return
        except Exception as e:
            fails += 1
            if fails < 5:
                time.sleep(10); continue
            log(f"  openfigi error {e}; stopping mapping"); return
        i += batch
        for c, item in zip(chunk, res):
            data = item.get("data") or []
            eq = [d for d in data if (d.get("securityType") or "").lower() in ("common stock", "reit", "adr", "etp", "mlp", "ltd part", "unit", "trust")] or data
            m[c] = (eq[0].get("ticker").replace("/", "-") if eq and eq[0].get("ticker") else None)
        if (i // batch) % 5 == 4:
            log(f"  openfigi {min(i + batch, len(cusips))}/{len(cusips)}")
        time.sleep(2.5)


# ---------------------------------------------------------------- prices

def monthly_prices(tickers: list[str], log=print) -> pd.DataFrame:
    """Wide frame: month-end index × ticker, adjusted close.  Cached per run in prices_monthly.csv."""
    import yfinance as yf
    cache = EDGAR / "prices_monthly.csv"
    have = pd.read_csv(cache, index_col=0, parse_dates=True) if cache.exists() else pd.DataFrame()
    todo = [t for t in dict.fromkeys(tickers) if t and t not in have.columns]
    frames = [have] if len(have) else []
    for i in range(0, len(todo), 150):
        chunk = todo[i:i + 150]
        try:
            df = yf.download(chunk, period="max", interval="1mo", auto_adjust=True, progress=False, threads=True, group_by="column")
        except Exception as e:
            log(f"  yfinance chunk failed: {e}"); continue
        close = df["Close"] if isinstance(df.columns, pd.MultiIndex) else df[["Close"]].rename(columns={"Close": chunk[0]})
        close.index = pd.DatetimeIndex(close.index).tz_localize(None).to_period("M").to_timestamp("M")
        close = close.groupby(level=0).last()
        frames.append(close)
        log(f"  prices {min(i + 150, len(todo))}/{len(todo)}")
    wide = pd.concat(frames, axis=1) if frames else pd.DataFrame()
    wide = wide.loc[:, ~wide.columns.duplicated()]
    # tickers that returned nothing are recorded as all-NaN columns so we don't refetch them
    for t in todo:
        if t not in wide.columns:
            wide[t] = np.nan
    wide.sort_index().to_csv(cache)
    return wide


# ---------------------------------------------------------------- clone

def clone_returns(h: pd.DataFrame, cmap: dict, prices: pd.DataFrame, detail: bool = False):
    """Monthly clone return, monthly priced coverage, and a per-period summary.
    With detail=True also returns (contrib, weights): month × ticker frames of w_i × r_i and of the
    beginning-of-month weight, so contributions add up exactly to each month's return."""
    rets = prices.pct_change(fill_method=None)
    last_full = (pd.Timestamp.today().to_period("M") - 1).to_timestamp("M")     # drop the partial current month
    prices = prices[prices.index <= last_full]; rets = rets[rets.index <= last_full]
    periods = sorted(h.period.unique())
    reb = {}
    for p in periods:
        filed = pd.Timestamp(h.loc[h.period == p, "filed"].iloc[0])
        reb[p] = filed.to_period("M").to_timestamp("M")      # rebalance at end of the filing month
    months = pd.date_range(min(reb.values()), prices.index.max(), freq="ME")
    out = pd.Series(np.nan, index=months); cov = pd.Series(np.nan, index=months)
    summary = []; contrib_rows = {}; weight_rows = {}
    order = sorted(periods, key=lambda p: reb[p])
    for k, p in enumerate(order):
        start = reb[p]
        end = reb[order[k + 1]] if k + 1 < len(order) else min(months[-1], start + pd.DateOffset(months=STALE_MONTHS))
        hp = h[h.period == p].copy()
        hp["ticker"] = hp.cusip.map(cmap)
        hp = hp[hp.ticker.notna() & hp.ticker.isin(prices.columns)]
        if start in prices.index:                       # priced = has a price at the rebalance date
            hp = hp[hp.ticker.map(prices.loc[start]).notna()]
        w = hp.groupby("ticker").weight.sum()
        priced_share = float(w.sum())
        summary.append(dict(period=p, rebalance=start.date(), n=int(len(w)), priced_share=priced_share))
        if w.empty:
            continue
        w = w / w.sum()
        cur = w.copy()
        for m in months[(months > start) & (months <= end)]:
            r = rets.loc[m, cur.index] if m in rets.index else pd.Series(np.nan, index=cur.index)
            ok = r.notna()
            c = float(cur[ok].sum()) * priced_share
            cov[m] = c
            if cur[ok].sum() <= 0 or c < MIN_COVERAGE or int(ok.sum()) < MIN_POSITIONS:
                continue
            wk = cur[ok] / cur[ok].sum()
            out[m] = float((wk * r[ok]).sum())
            if detail:
                contrib_rows[m] = (wk * r[ok]).to_dict(); weight_rows[m] = wk.to_dict()
            cur = cur[ok] * (1 + r[ok]); cur = cur / cur.sum()
    if detail:
        contrib = pd.DataFrame.from_dict(contrib_rows, orient="index").sort_index()
        weights = pd.DataFrame.from_dict(weight_rows, orient="index").sort_index()
        return out, cov, pd.DataFrame(summary), contrib, weights
    return out, cov, pd.DataFrame(summary)


def contribution_table(contrib: pd.DataFrame, weights: pd.DataFrame, months: pd.Index, names: dict,
                       rm: pd.Series | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-holding contribution over `months` (the scored window).

    contribution  = Σ_t w_it · r_it                (adds up to the clone's return)
    active        = Σ_t w_it · (r_it − r_mt)       (adds up to the clone's excess over the market —
                                                    a stock that merely matched the market contributes 0)
    share_of_outperformance = active ÷ Σ positive actives; cumulative share in rank order.
    Also returns a long timeline (ticker, month, weight) for the names that matter."""
    c = contrib.reindex(months).fillna(0.0); w = weights.reindex(months)
    tot = c.sum()
    if rm is not None:
        rmm = rm.reindex(months).fillna(0.0)
        act = (c - w.fillna(0.0).mul(rmm, axis=0)).sum()      # w·r − w·rm
    else:
        act = tot.copy()
    held = w.notna() & (w > 0)
    df = pd.DataFrame({"ticker": tot.index, "contribution": tot.values, "active": act.reindex(tot.index).values,
                       "months_held": held.sum().reindex(tot.index).values,
                       "avg_weight_when_held": w.where(held).mean().reindex(tot.index).values})
    first = {t: held.index[held[t]].min() for t in tot.index if held[t].any()}
    last = {t: held.index[held[t]].max() for t in tot.index if held[t].any()}
    df["first_held"] = df.ticker.map(lambda t: str(first[t].date()) if t in first else "")
    df["last_held"] = df.ticker.map(lambda t: str(last[t].date()) if t in last else "")
    df["name"] = df.ticker.map(names).fillna(df.ticker)
    df = df.sort_values("active", ascending=False).reset_index(drop=True)
    pos = df.active[df.active > 0].sum()
    df["share_of_outperformance"] = np.where(df.active > 0, df.active / pos if pos > 0 else np.nan, np.nan)
    df["cum_share_of_outperformance"] = df.share_of_outperformance.fillna(0).cumsum().where(df.active > 0)
    df["rank"] = np.arange(1, len(df) + 1)
    keep = list(df.head(20).ticker) + list(df.tail(6).ticker)
    tl = w[[t for t in keep if t in w.columns]].where(held).stack().reset_index()
    tl.columns = ["month", "ticker", "weight"]
    tl["month"] = pd.to_datetime(tl["month"]).dt.strftime("%Y-%m")
    return df, tl


def write_fund_dataset(fund: dict, ret: pd.Series, cov: pd.Series, summary: pd.DataFrame, out_dir: Path,
                       contrib: pd.DataFrame | None = None, weights: pd.DataFrame | None = None, names: dict | None = None,
                       rm: pd.Series | None = None) -> dict | None:
    from .fund_universe import STYLES
    # keep only the longest gap-free stretch: a series is never chained across a hole
    valid = ret.notna()
    best, cur_start, cur_len, best_start = 0, None, 0, None
    for i, ok in enumerate(valid.values):
        if ok:
            if cur_start is None: cur_start = i
            cur_len += 1
            if cur_len > best: best, best_start = cur_len, cur_start
        else:
            cur_start, cur_len = None, 0
    total_valid = int(valid.sum())
    ret = ret.iloc[best_start:best_start + best] if best else ret.iloc[0:0]
    r = ret.dropna()
    if len(r) < 36:
        meta = dict(slug=fund["slug"], name=fund["name"], manager=fund["manager"], style=fund["style"],
                    style_name=STYLES[fund["style"]][0], style_note=STYLES[fund["style"]][1],
                    months=int(len(r)), status="insufficient",
                    note=(f"only {len(r)} months of clone history (need 36)" if total_valid < 36 else
                          f"longest gap-free stretch is {len(r)} months ({total_valid} scattered valid months); too few priced holdings"),
                    first=str(r.index.min().date()) if len(r) else None, last=str(r.index.max().date()) if len(r) else None,
                    filings=int(len(summary)))
        out_dir.mkdir(parents=True, exist_ok=True); (out_dir / "meta.json").write_text(json.dumps(meta)); return meta
    # contiguous from the first non-NaN month; a NaN inside becomes a gap the pipeline reports honestly
    acct = re.sub(r"[^A-Z0-9]", "_", fund["slug"].upper())
    val = 1_000_000.0
    st_rows, fl_rows = [], []
    first = r.index.min()
    st_rows.append(dict(statement_id=f"{acct}_{first.to_period('M')}", account_id=acct, custodian="SEC 13F clone (EDGAR + Yahoo Finance)",
                        period_start=first.to_period("M").start_time.date(), period_end=first.date(), ending_value=round(val, 2),
                        source_file="hedge13f", source_pages="", notes="notional $1m invested at the first rebalance"))
    fl_rows.append(dict(account_id=acct, date=first.date(), amount=round(val, 2), flow_type="deposit",
                        description="notional $1m at first 13F rebalance", source_statement_id=f"{acct}_{first.to_period('M')}"))
    for m in ret.index[ret.index > first]:
        x = ret[m]
        if pd.isna(x):
            continue                       # hole -> no statement that month; Phase 1 reports the gap
        val = val * (1 + x)
        st_rows.append(dict(statement_id=f"{acct}_{m.to_period('M')}", account_id=acct, custodian="SEC 13F clone (EDGAR + Yahoo Finance)",
                            period_start=m.to_period("M").start_time.date(), period_end=m.date(), ending_value=round(val, 2),
                            source_file="hedge13f", source_pages="", notes=f"priced coverage {cov[m]:.0%}"))
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = ["statement_id", "account_id", "custodian", "period_start", "period_end", "ending_value",
            "beginning_value", "stated_deposits", "stated_withdrawals", "stated_income", "stated_fees",
            "stated_pnl", "stated_return_pct", "source_file", "source_pages", "notes"]
    pd.DataFrame(st_rows).reindex(columns=cols).to_csv(out_dir / "statements.csv", index=False)
    pd.DataFrame(fl_rows).to_csv(out_dir / "flows.csv", index=False)
    pd.DataFrame(columns=["statement_id", "account_id", "as_of_date", "identifier", "description", "asset_class", "quantity",
                          "price", "market_value", "weight_pct"]).to_csv(out_dir / "positions.csv", index=False)
    pd.DataFrame([dict(account_id=acct, label=f"{fund['name']} — 13F clone", owner_type="principal", discretionary="Y",
                       strategy="default", benchmark="US_MKT", notes="13F long-only clone; not the fund's return")]).to_csv(out_dir / "accounts.csv", index=False)
    summary.to_csv(out_dir / "filings.csv", index=False)
    if contrib is not None and weights is not None:
        ct, tl = contribution_table(contrib, weights, ret.index[ret.index > first], names or {}, rm)
        ct.to_csv(out_dir / "contributions.csv", index=False); tl.to_csv(out_dir / "timeline.csv", index=False)
    meta = dict(slug=fund["slug"], name=fund["name"], manager=fund["manager"], style=fund["style"],
                style_name=STYLES[fund["style"]][0], style_note=STYLES[fund["style"]][1],
                months=int(len(st_rows)), status="ok", first=str(first.date()), last=str(ret.dropna().index.max().date()),
                filings=int(len(summary)), avg_coverage=float(cov.reindex(r.index).dropna().mean()) if cov.notna().any() else None,
                avg_positions=float(summary.n.mean()) if len(summary) else None, holes=0, dropped_months=int(total_valid - len(r)),
                last_filing=str(summary.period.max()) if len(summary) else None,
                ciks=[f["cik"] for f in fund["filers"]])
    (out_dir / "meta.json").write_text(json.dumps(meta))
    return meta


def build_all(funds: list[dict], out_root: Path, log=print, only: list[str] | None = None, figi: bool = True) -> list[dict]:
    """Stage 1: holdings for every fund (cached).  Stage 2: one CUSIP map + one price pull for the
    whole universe.  Stage 3: clone series and datasets."""
    sel = [f for f in funds if not only or f["slug"] in only]
    H = {}
    for i, f in enumerate(sel):
        t0 = time.time()
        H[f["slug"]] = fund_holdings(f, log)
        log(f"[{i + 1}/{len(sel)}] {f['slug']}: {H[f['slug']].period.nunique()} filings, {len(H[f['slug']])} rows ({time.time() - t0:.0f}s)")
    allh = pd.concat([h for h in H.values() if len(h)], ignore_index=True)
    log(f"universe: {allh.cusip.nunique()} unique CUSIPs across {len(sel)} funds")
    cmap = map_cusips(allh, log, figi=figi)
    tickers = sorted({t for t in cmap.values() if t})
    log(f"mapped {len(tickers)} tickers; fetching prices")
    prices = monthly_prices(tickers, log)
    try:                                   # US market monthly return, for contribution to outperformance
        from .reference import load_all
        rm = load_all()["US_MKT"]; rm.index = pd.DatetimeIndex(rm.index).to_period("M").to_timestamp("M")
    except Exception as e:
        log(f"  market series unavailable ({e}); active contributions fall back to gross"); rm = None
    metas = []
    for f in sel:
        h = H[f["slug"]]
        if h.empty:
            log(f"  {f['slug']}: no structured filings"); continue
        ret, cov, summary, contrib, weights = clone_returns(h, cmap, prices, detail=True)
        names = {}
        for cusip, nm in zip(h.cusip, h.name):
            t = cmap.get(cusip)
            if t and t not in names:
                names[t] = nm.title()
        meta = write_fund_dataset(f, ret, cov, summary, out_root / f["slug"], contrib, weights, names, rm)
        if meta:
            metas.append(meta)
            log(f"  {f['slug']}: {meta['status']} {meta.get('months', 0)} months, coverage {meta.get('avg_coverage') or 0:.0%}")
    return metas
