"""Portable copy of the reference caches, small enough to commit.

The raw caches under data/reference/ (EDGAR 13F JSON, submissions, prices,
CUSIP map, Ken French zips) are ~450 MB and gitignored, so a fresh clone
(a cloud session, a new laptop) cannot run the 13F work without ~1.5 h of
EDGAR pulls.  `pack` writes a ~15 MB bundle to data/reference/compact/ —
which IS committed — and `unpack` rebuilds the raw layout from it, so every
existing loader works unchanged.

Only managers whose clone is meaningful are packed (multi-strategy, macro
and market-making books are 80% of the rows and are never scored); their
holdings still fetch from EDGAR on demand.

    python -m trackrecord compact-pack      # laptop, after funds-build
    python -m trackrecord compact-unpack    # fresh clone; no-op if raw caches exist
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from .fund_universe import STYLES
from .reference import FILES as KF_FILES

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "data" / "reference"
EDGAR = REF / "edgar"
COMPACT = REF / "compact"
SKIP_STYLES = {"multi", "macro", "mm"}
XML_ERA = "2013-06-30"
HOLDING_COLS = ["slug", "period", "filed", "accession", "cik", "cusip", "name", "value", "shares", "putcall", "cls"]


def pack(log=print) -> dict:
    COMPACT.mkdir(parents=True, exist_ok=True)
    funds = json.loads((EDGAR / "funds.json").read_text())
    cik2slug = {str(int(fl["cik"])): f["slug"] for f in funds for fl in f["filers"]}
    style = {f["slug"]: f["style"] for f in funds}
    subs, rows, skipped = {}, [], 0
    for sub in sorted((EDGAR / "submissions").glob("*.json")):
        cik = str(int(sub.stem))
        filings = json.loads(sub.read_text())
        subs[cik] = filings
        slug = cik2slug.get(cik)
        if not slug or style[slug] in SKIP_STYLES:
            continue
        for fl in filings:
            if fl["period"] < XML_ERA:
                continue
            acc = fl["accession"].replace("-", "")
            p = EDGAR / "13f" / cik / f"{acc}.json"
            if not p.exists():
                continue
            h = json.loads(p.read_text())
            if not h:
                continue
            for r in h:
                rows.append((slug, fl["period"], fl["filed"], fl["accession"], cik, r["cusip"], r["name"],
                             r["value"], r["shares"], r.get("putcall", ""), r.get("cls", "")))
    df = pd.DataFrame(rows, columns=HOLDING_COLS)
    df.to_csv(COMPACT / "holdings13f.csv.gz", index=False, compression="gzip")
    (COMPACT / "submissions.json").write_text(json.dumps(subs))
    shutil.copy(EDGAR / "funds.json", COMPACT / "funds.json")
    shutil.copy(EDGAR / "cusip_map.json", COMPACT / "cusip_map.json")
    prices = pd.read_csv(EDGAR / "prices_monthly.csv", index_col=0)
    prices.round(4).to_csv(COMPACT / "prices_monthly.csv.gz", compression="gzip")
    if (EDGAR / "prices_close_monthly.csv").exists():
        pd.read_csv(EDGAR / "prices_close_monthly.csv", index_col=0).round(4).to_csv(COMPACT / "prices_close_monthly.csv.gz", compression="gzip")
    kf = COMPACT / "kenfrench"; kf.mkdir(exist_ok=True)
    for zip_name, _ in KF_FILES.values():
        if (REF / zip_name).exists():
            shutil.copy(REF / zip_name, kf / zip_name)
    info = dict(filings=int(df.accession.nunique()), rows=int(len(df)), managers=int(df.slug.nunique()),
                prices=list(prices.shape), skipped_styles=sorted(SKIP_STYLES))
    (COMPACT / "manifest.json").write_text(json.dumps(info, indent=1))
    log(f"packed {info['filings']} filings / {info['rows']} rows for {info['managers']} managers; "
        f"prices {prices.shape[0]}×{prices.shape[1]}; -> {COMPACT}")
    return info


def unpack(log=print, force: bool = False) -> bool:
    """Rebuild the raw cache layout from the bundle.  Returns False if there is nothing to do."""
    if not (COMPACT / "manifest.json").exists():
        return False
    if (EDGAR / "prices_monthly.csv").exists() and (EDGAR / "cusip_map.json").exists() and not force:
        return False
    EDGAR.mkdir(parents=True, exist_ok=True)
    (EDGAR / "submissions").mkdir(exist_ok=True)
    for cik, filings in json.loads((COMPACT / "submissions.json").read_text()).items():
        (EDGAR / "submissions" / f"{int(cik):010d}.json").write_text(json.dumps(filings))
    shutil.copy(COMPACT / "funds.json", EDGAR / "funds.json")
    shutil.copy(COMPACT / "cusip_map.json", EDGAR / "cusip_map.json")
    pd.read_csv(COMPACT / "prices_monthly.csv.gz", index_col=0).to_csv(EDGAR / "prices_monthly.csv")
    if (COMPACT / "prices_close_monthly.csv.gz").exists():
        pd.read_csv(COMPACT / "prices_close_monthly.csv.gz", index_col=0).to_csv(EDGAR / "prices_close_monthly.csv")
    df = pd.read_csv(COMPACT / "holdings13f.csv.gz", dtype={"cusip": str, "cik": str, "putcall": str, "cls": str}, keep_default_na=False)
    n = 0
    for (cik, acc), g in df.groupby(["cik", "accession"], sort=False):
        d = EDGAR / "13f" / str(int(cik)); d.mkdir(parents=True, exist_ok=True)
        recs = [dict(cusip=r.cusip, name=r.name, value=float(r.value), shares=float(r.shares), putcall=r.putcall, cls=r.cls)
                for r in g.itertuples(index=False)]
        (d / f"{acc.replace('-', '')}.json").write_text(json.dumps(recs)); n += 1
    for z in (COMPACT / "kenfrench").glob("*.zip"):
        if not (REF / z.name).exists():
            shutil.copy(z, REF / z.name)
    log(f"unpacked {n} filings, prices, CUSIP map, submissions and factor zips into {REF}")
    return True


def holdings_frame() -> pd.DataFrame:
    """The packed holdings as one long frame (works with or without the raw cache)."""
    return pd.read_csv(COMPACT / "holdings13f.csv.gz", dtype={"cusip": str, "cik": str, "putcall": str, "cls": str}, keep_default_na=False)
