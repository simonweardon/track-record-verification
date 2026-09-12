"""Any listed vehicle as a one-account placeholder: mutual funds, ETFs, holding
companies.  Monthly adjusted closes (distributions reinvested) via yfinance, in
the pipeline's own statement format — the same treatment as the Berkshire set.

A price feed is not a custodian statement, so every period is honestly
UNVERIFIED; the point is the return series and the Phase 2–4 statistics.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

TICKER_RE = re.compile(r"^[A-Z0-9][A-Z0-9.\-]{0,11}$")

# Famous managers whose vehicle is listed — real monthly public records.
FAMOUS = [
    ("BRK-A", "Warren Buffett — Berkshire Hathaway"),
    ("FCNTX", "Will Danoff — Fidelity Contrafund (since 1990)"),
    ("FMAGX", "Fidelity Magellan (Peter Lynch era 1977–90, then successors)"),
    ("SEQUX", "Sequoia Fund (Ruane, Cunniff)"),
    ("DODGX", "Dodge & Cox Stock"),
    ("VWELX", "Vanguard Wellington (balanced, since 1929)"),
    ("ARKK", "Cathie Wood — ARK Innovation"),
    ("FAIRX", "Bruce Berkowitz — Fairholme"),
    ("MKL", "Tom Gayner — Markel"),
    ("FRFHF", "Prem Watsa — Fairfax Financial"),
]
NOT_PUBLIC = "Tepper (Appaloosa), Gavin Baker (Atreides), Druckenmiller, Einhorn, Ackman's private funds"


def valid(ticker: str) -> bool:
    return bool(TICKER_RE.match(ticker or ""))


def fetch(ticker: str, out_csv: str | Path) -> dict:
    """Download monthly adjusted history; write CSV + meta.json (name, currency)."""
    import yfinance as yf
    t = yf.Ticker(ticker)
    h = t.history(period="max", interval="1mo", auto_adjust=True)
    if h is None or h.empty or "Close" not in h:
        raise ValueError(f"no price history for {ticker!r} — check the symbol (Yahoo Finance format, e.g. BRK-A, FCNTX)")
    name, currency = ticker, "USD"
    try:                                    # .info is slow/flaky; a name is nice-to-have
        info = t.info or {}
        name = info.get("longName") or info.get("shortName") or ticker
        currency = info.get("currency") or currency
    except Exception:
        pass
    out = Path(out_csv); out.parent.mkdir(parents=True, exist_ok=True)
    h.to_csv(out)
    meta = dict(ticker=ticker, name=name, currency=currency, rows=int(len(h)),
                first=str(h.index.min().date()), last=str(h.index.max().date()))
    out.with_suffix(".json").write_text(json.dumps(meta))
    return meta


def build(ticker: str, raw_csv: str | Path, out_dir: str | Path, shares: float = 100.0) -> Path:
    raw = pd.read_csv(raw_csv)
    raw["Date"] = pd.to_datetime(raw["Date"], utc=True).dt.tz_convert(None).dt.normalize()
    raw = raw.dropna(subset=["Close"]).sort_values("Date")
    raw = raw[raw["Close"] > 0]
    raw["month"] = raw["Date"].dt.to_period("M")
    raw = raw.drop_duplicates("month", keep="last")
    raw = raw[raw["month"] <= pd.Timestamp.today().to_period("M") - 1]      # drop the partial current month
    if len(raw) < 36:
        raise ValueError(f"{ticker}: only {len(raw)} months of history — need at least 36")
    meta_p = Path(raw_csv).with_suffix(".json")
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {"name": ticker}
    acct = re.sub(r"[^A-Z0-9]", "_", ticker.upper())
    # scale so the first month-end value is a round number of shares' worth
    st_rows, fl_rows = [], []
    for i, r in enumerate(raw.itertuples(index=False)):
        p_start = r.month.start_time.normalize(); p_end = r.month.end_time.normalize()
        end = round(shares * float(r.Close), 2)
        st_rows.append(dict(statement_id=f"{acct}_{r.month}", account_id=acct,
                            custodian="Yahoo Finance (public price series, distributions reinvested)",
                            period_start=p_start.date(), period_end=p_end.date(), ending_value=end,
                            source_file=Path(raw_csv).name, source_pages="",
                            notes="public monthly adjusted close; no statement to reconcile against"))
        if i == 0:
            fl_rows.append(dict(account_id=acct, date=p_end.date(), amount=end, flow_type="deposit",
                                description=f"notional purchase of {shares:g} {ticker} at month-end", source_statement_id=f"{acct}_{r.month}"))
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    cols = ["statement_id", "account_id", "custodian", "period_start", "period_end", "ending_value",
            "beginning_value", "stated_deposits", "stated_withdrawals", "stated_income", "stated_fees",
            "stated_pnl", "stated_return_pct", "source_file", "source_pages", "notes"]
    pd.DataFrame(st_rows).reindex(columns=cols).to_csv(out / "statements.csv", index=False)
    pd.DataFrame(fl_rows).to_csv(out / "flows.csv", index=False)
    pd.DataFrame(columns=["statement_id", "account_id", "as_of_date", "identifier", "description",
                          "asset_class", "quantity", "price", "market_value", "weight_pct"]).to_csv(out / "positions.csv", index=False)
    pd.DataFrame([dict(account_id=acct, label=f"{meta.get('name', ticker)} ({ticker})", owner_type="principal",
                       discretionary="Y", strategy="default", benchmark="US_MKT",
                       notes="public price series via yfinance; not the record under verification")]).to_csv(out / "accounts.csv", index=False)
    (out / "meta.json").write_text(json.dumps(meta))
    return out
