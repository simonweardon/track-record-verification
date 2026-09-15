"""Company fundamentals from SEC XBRL "company facts" — free, point-in-time, no vendor.

For every ticker in the research universe: the quarterly/annual facts a value-and-quality
signal needs, taken from each company's own filings via
https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json, and kept **as filed** — each
observation carries the period end and the filing date, so a signal can be built without
look-ahead (use a fact only after its `filed` date).

Facts kept (us-gaap unless noted; first tag that exists wins):
    equity      StockholdersEquity | StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest
    net_income  NetIncomeLoss
    revenue     Revenues | RevenueFromContractWithCustomerExcludingAssessedTax | SalesRevenueNet
    op_cf       NetCashProvidedByUsedInOperatingActivities
    assets      Assets
    shares      dei:EntityCommonStockSharesOutstanding | CommonStockSharesOutstanding

Output: data/reference/compact/fundamentals.csv.gz (ticker, cik, fact, end, filed, fy, fp, form, value).
    python -m trackrecord fundamentals --contact "Name email"
"""
from __future__ import annotations

import gzip
import json
import time
import urllib.request
from pathlib import Path

import pandas as pd

from .compact import COMPACT, EDGAR

FACTS = {
    "equity": [("us-gaap", "StockholdersEquity"), ("us-gaap", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest")],
    "net_income": [("us-gaap", "NetIncomeLoss")],
    "revenue": [("us-gaap", "Revenues"), ("us-gaap", "RevenueFromContractWithCustomerExcludingAssessedTax"), ("us-gaap", "SalesRevenueNet")],
    "op_cf": [("us-gaap", "NetCashProvidedByUsedInOperatingActivities")],
    "assets": [("us-gaap", "Assets")],
    "shares": [("dei", "EntityCommonStockSharesOutstanding"), ("us-gaap", "CommonStockSharesOutstanding")],
}
OUT = COMPACT / "fundamentals.csv.gz"
CACHE = EDGAR / "companyfacts"


def ticker_ciks() -> dict[str, int]:
    ct = json.loads((EDGAR / "company_tickers.json").read_text())
    return {v["ticker"].upper().replace(".", "-"): int(v["cik_str"]) for v in ct.values()}


def _get(url: str, ua: dict, retries: int = 3) -> bytes | None:
    for k in range(retries):
        try:
            req = urllib.request.Request(url, headers=ua)
            with urllib.request.urlopen(req, timeout=40) as r:
                raw = r.read()
                return gzip.decompress(raw) if r.headers.get("Content-Encoding") == "gzip" else raw
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(3 * (k + 1))
        except Exception:
            time.sleep(2 * (k + 1))
    return None


def company_facts(cik: int, ua: dict) -> dict | None:
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{cik:010d}.json"
    if f.exists():
        return json.loads(f.read_text())
    raw = _get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json", ua)
    if raw is None:
        f.write_text("null"); return None
    f.write_bytes(raw)
    return json.loads(raw)


def extract(ticker: str, cik: int, facts: dict) -> list[dict]:
    rows = []
    fx = facts.get("facts", {})
    for name, tags in FACTS.items():
        for taxonomy, tag in tags:
            node = fx.get(taxonomy, {}).get(tag)
            if not node:
                continue
            units = node.get("units", {})
            series = units.get("USD") or units.get("shares") or next(iter(units.values()), [])
            for o in series:
                if o.get("form") not in ("10-K", "10-Q", "10-K/A", "10-Q/A", "20-F", "40-F"):
                    continue
                rows.append(dict(ticker=ticker, cik=cik, fact=name, end=o.get("end"), filed=o.get("filed"), fy=o.get("fy"), fp=o.get("fp"),
                                 form=o.get("form"), start=o.get("start"), value=o.get("val")))
            break                                   # first tag that exists wins
    return rows


def build(tickers: list[str], contact: str, log=print) -> pd.DataFrame:
    ua = {"User-Agent": contact, "Accept-Encoding": "gzip, deflate"}
    ciks = ticker_ciks()
    rows, miss, t0, last = [], [], time.time(), 0.0
    for i, t in enumerate(tickers):
        cik = ciks.get(t.upper())
        if cik is None:
            miss.append(t); continue
        wait = 0.12 - (time.time() - last)
        if wait > 0:
            time.sleep(wait)
        last = time.time()
        facts = company_facts(cik, ua)
        if facts:
            rows.extend(extract(t, cik, facts))
        else:
            miss.append(t)
        if (i + 1) % 100 == 0:
            log(f"  fundamentals {i + 1}/{len(tickers)} ({time.time() - t0:.0f}s)")
    df = pd.DataFrame(rows).drop_duplicates(["ticker", "fact", "end", "filed", "start"])
    df.to_csv(OUT, index=False, compression="gzip")
    log(f"fundamentals: {df.ticker.nunique()} tickers, {len(df):,} facts; {len(miss)} without a CIK or facts -> {OUT}")
    return df


def load() -> pd.DataFrame:
    return pd.read_csv(OUT, dtype={"fy": "float", "fp": str, "form": str}, parse_dates=["end", "filed"]) if OUT.exists() else pd.DataFrame()


def universe_tickers() -> list[str]:
    """Every ticker in the ≥ 5-holder research universe (the sectors file), else every priced 13F name."""
    sec = COMPACT / "sectors.csv"
    if sec.exists():
        return sorted(pd.read_csv(sec, dtype=str).ticker.dropna().unique())
    from .signals13f import load_books
    b = load_books(log=lambda *a: None)
    return sorted(b.ticker.dropna().unique())
