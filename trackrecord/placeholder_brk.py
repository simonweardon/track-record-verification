"""Second placeholder: Berkshire Hathaway Class A (BRK-A) monthly closes, as one
'account' in the pipeline's own statement format.

Why this series: it is real, famous, 40+ years long, MONTHLY (so the parts of
Phase 4 that annual data cannot support get exercised), pays no dividend (price
return = total return), and has a published academic baseline to sanity-check
against (Frazzini, Kabiller & Pedersen, "Buffett's Alpha", FAJ 2018).

What it is not: it is the stock, not Buffett's public-equity book — it carries the
operating businesses and ~1.6x insurance-float leverage — and a price feed is
not a custodian statement, so Phase 1 correctly marks every period UNVERIFIED.
Nothing here is the record under verification.

Source file: data/reference/brk-a_monthly_raw.csv (yfinance, interval=1mo).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

SHARES = 100          # a $139,500 'account' at the Jan-1985 close


def build(raw_csv: str | Path, out_dir: str | Path, through: str | None = None) -> Path:
    raw = pd.read_csv(raw_csv)
    raw["Date"] = pd.to_datetime(raw["Date"], utc=True).dt.tz_convert(None).dt.normalize()
    raw = raw.dropna(subset=["Close"]).sort_values("Date")
    # yfinance monthly bars are labelled by month start; Close is that month's last close
    raw["month"] = raw["Date"].dt.to_period("M")
    raw = raw.drop_duplicates("month", keep="last")
    last_full = pd.Timestamp.today().to_period("M") - 1
    raw = raw[raw["month"] <= (pd.Period(through, "M") if through else last_full)]

    acct = "BRK_A"
    st_rows, fl_rows = [], []
    for i, r in enumerate(raw.itertuples(index=False)):
        p_start = r.month.start_time.normalize(); p_end = r.month.end_time.normalize()
        end = round(SHARES * float(r.Close), 2)
        st_rows.append(dict(statement_id=f"{acct}_{r.month}", account_id=acct,
                            custodian="Yahoo Finance (public price series)",
                            period_start=p_start.date(), period_end=p_end.date(), ending_value=end,
                            source_file="brk-a_monthly_raw.csv", source_pages="",
                            notes="public monthly close x 100 shares; no statement to reconcile against"))
        if i == 0:
            fl_rows.append(dict(account_id=acct, date=p_end.date(), amount=end, flow_type="deposit",
                                description="notional purchase of 100 BRK.A at month-end close",
                                source_statement_id=f"{acct}_{r.month}"))
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    cols = ["statement_id", "account_id", "custodian", "period_start", "period_end", "ending_value",
            "beginning_value", "stated_deposits", "stated_withdrawals", "stated_income", "stated_fees",
            "stated_pnl", "stated_return_pct", "source_file", "source_pages", "notes"]
    pd.DataFrame(st_rows).reindex(columns=cols).to_csv(out / "statements.csv", index=False)
    pd.DataFrame(fl_rows).to_csv(out / "flows.csv", index=False)
    pd.DataFrame(columns=["statement_id", "account_id", "as_of_date", "identifier", "description",
                          "asset_class", "quantity", "price", "market_value", "weight_pct"]).to_csv(out / "positions.csv", index=False)
    pd.DataFrame([dict(account_id=acct, label="Berkshire Hathaway Class A (public price series)",
                       owner_type="principal", discretionary="Y", strategy="default", benchmark="US_MKT",
                       notes="placeholder; BRK-A monthly closes via yfinance; not the record under verification")]
                 ).to_csv(out / "accounts.csv", index=False)
    return out
