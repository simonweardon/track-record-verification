"""Turn an uploaded returns / values / statement file into a pipeline dataset.

Three accepted shapes (CSV or XLSX):

  returns   date, return              monthly or annual; return as decimal (0.012) or percent (1.2)
  values    date, value[, flow]       ending values; flow +deposit / −withdrawal on that date
  template  statements.csv, flows.csv, positions.csv[, accounts.csv]   the pipeline's own format

`returns` and `values` become one account with a notional or actual value path; nothing on
them can be reconciled, so the pipeline marks every period unverified — which is the truth.
Of these three the template shape is the only one that can reach "verified"; statement PDFs
(see pdfstatements.py) can too, because they print a beginning value to chain against.
"""
from __future__ import annotations

import io
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

MAX_BYTES = 5 * 1024 * 1024          # per file
MAX_TOTAL = 40 * 1024 * 1024         # per upload, so a long run of monthly statements fits


class UploadError(Exception):
    pass


def _read_table(name: str, data: bytes) -> pd.DataFrame:
    if len(data) > MAX_BYTES:
        raise UploadError(f"{name}: file is larger than 5 MB")
    low = name.lower()
    try:
        if low.endswith((".xlsx", ".xlsm", ".xls")):
            return pd.read_excel(io.BytesIO(data))
        return pd.read_csv(io.BytesIO(data), dtype=str, keep_default_na=False, skipinitialspace=True)
    except Exception as e:
        raise UploadError(f"{name}: could not be read as a table ({e})")


def _norm(c) -> str:
    return re.sub(r"[^a-z]", "", str(c).lower())


def _find_col(df: pd.DataFrame, *cands) -> str | None:
    """The column a candidate name refers to, exactly first, then as part of a longer name.

    Broker exports rarely call a column just "date": Robinhood writes "Activity Date",
    others "As Of Date" or "Month End". A candidate of four letters or more is allowed to
    match inside a longer heading so those files load without being edited first.
    """
    cols = {_norm(c): c for c in df.columns}
    for c in cands:
        if c in cols:
            return cols[c]
    for c in cands:
        if len(c) < 4:
            continue
        for k, orig in cols.items():
            if c in k:
                return orig
    return None


# columns that say "this file is a list of transactions", not a history of the account
TXN_COLS = {"transcode", "activitydate", "settledate", "tradedate", "processdate", "action",
            "instrument", "symbol", "quantity", "price", "amount", "description", "commission"}


def _transaction_export(df: pd.DataFrame) -> str | None:
    """A message explaining why a transaction list cannot become a track record, or None."""
    n = {_norm(c) for c in df.columns}
    if len(n & TXN_COLS) < 4:
        return None
    who = "Robinhood's transaction export looks exactly like this. " if {"activitydate", "transcode"} <= n else ""
    return ("this file lists individual transactions — each trade, dividend, fee and transfer — rather than "
            "what the account was worth over time. " + who + "A transaction list never prints the account's "
            "value at the end of each month, and that value is what a return is computed from, so a track "
            "record cannot be built from it. Upload the monthly statements themselves instead, with the "
            "Statement PDFs option: they print the opening and closing balance of every month. In Robinhood "
            "they are under Account, then Menu, then Statements.")


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(r"[,$%\s]", "", regex=True).replace({"": np.nan, "-": np.nan}), errors="coerce")


def _dates(s: pd.Series) -> pd.Series:
    d = pd.to_datetime(s.astype(str).str.strip(), errors="coerce")
    if d.isna().mean() > 0.2:
        raise UploadError("the date column could not be parsed — use YYYY-MM-DD, YYYY-MM or YYYY")
    return d


def _grid(dates: pd.Series) -> str:
    gaps = dates.sort_values().diff().dt.days.dropna()
    med = float(gaps.median()) if len(gaps) else 365
    if med < 45:
        return "M"
    if med > 300:
        return "A"
    raise UploadError("dates look neither monthly nor annual (median spacing %.0f days)" % med)


def _cols():
    return ["statement_id", "account_id", "custodian", "period_start", "period_end", "ending_value",
            "beginning_value", "stated_deposits", "stated_withdrawals", "stated_income", "stated_fees",
            "stated_pnl", "stated_return_pct", "source_file", "source_pages", "notes"]


def _period(d: pd.Timestamp, grid: str):
    if grid == "M":
        return d.to_period("M").start_time.normalize().date(), d.to_period("M").end_time.normalize().date()
    return pd.Timestamp(d.year, 1, 1).date(), pd.Timestamp(d.year, 12, 31).date()


def from_returns(name: str, data: bytes, out: Path, label: str) -> dict:
    df = _read_table(name, data)
    dc, rc = _find_col(df, "date", "month", "period", "year", "asof"), _find_col(df, "return", "ret", "monthlyreturn", "annualreturn", "netreturn", "grossreturn", "pct", "performance")
    if dc is None or rc is None:
        why = _transaction_export(df)
        raise UploadError(f"{name}: {why}" if why else
                          f"{name}: need a date column and a return column (found: {', '.join(df.columns)})")
    d = pd.DataFrame({"date": _dates(df[dc]), "r": _num(df[rc])}).dropna().sort_values("date")
    if len(d) < 12:
        raise UploadError(f"{name}: only {len(d)} usable rows — need at least 12")
    if d.r.abs().max() > 1.5:          # percent, not decimal
        d["r"] = d.r / 100.0
    grid = _grid(d.date)
    acct = "UPLOAD"
    val = 1_000_000.0
    st, fl = [], []
    first_start, first_end = _period(d.date.iloc[0], grid)
    prev_start = pd.Timestamp(first_start) - (pd.DateOffset(months=1) if grid == "M" else pd.DateOffset(years=1))
    ps, pe = _period(prev_start, grid)
    st.append(dict(statement_id=f"{acct}_{pe}", account_id=acct, custodian="uploaded return series", period_start=ps, period_end=pe,
                   ending_value=round(val, 2), source_file=name, notes="notional $1m at the start of the series"))
    fl.append(dict(account_id=acct, date=pe, amount=round(val, 2), flow_type="deposit", description="notional start", source_statement_id=f"{acct}_{pe}"))
    for _, x in d.iterrows():
        val *= (1 + float(x.r))
        ps, pe = _period(x.date, grid)
        st.append(dict(statement_id=f"{acct}_{pe}", account_id=acct, custodian="uploaded return series", period_start=ps, period_end=pe,
                       ending_value=round(val, 2), stated_return_pct=round(float(x.r) * 100, 6), source_file=name, notes=""))
    _write(out, st, fl, acct, label, "uploaded return series (nothing to reconcile against)")
    return dict(grid=grid, periods=len(d), first=str(d.date.iloc[0].date()), last=str(d.date.iloc[-1].date()), shape="returns")


def from_values(name: str, data: bytes, out: Path, label: str) -> dict:
    df = _read_table(name, data)
    dc = _find_col(df, "date", "month", "period", "asof", "year")
    vc = _find_col(df, "value", "endingvalue", "closingvalue", "closingbalance", "endingbalance", "balance", "nav",
                   "marketvalue", "totalvalue", "portfoliovalue", "accountvalue", "equity")
    fc = _find_col(df, "flow", "netflow", "cashflow", "netdeposits", "deposit", "contribution", "depositwithdrawal", "flows")
    wc = _find_col(df, "withdrawal", "withdrawals", "redemption")
    if dc is None or vc is None:
        why = _transaction_export(df)
        raise UploadError(f"{name}: {why}" if why else
                          f"{name}: need a date column and a value column (found: {', '.join(df.columns)})")
    d = pd.DataFrame({"date": _dates(df[dc]), "v": _num(df[vc])})
    d["f"] = _num(df[fc]) if fc else 0.0
    if wc:
        d["f"] = d.f.fillna(0) - _num(df[wc]).fillna(0)
    d = d.dropna(subset=["date", "v"]).sort_values("date")
    if len(d) < 12:
        raise UploadError(f"{name}: only {len(d)} usable rows — need at least 12")
    grid = _grid(d.date)
    acct = "UPLOAD"
    st, fl = [], []
    for i, x in enumerate(d.itertuples(index=False)):
        ps, pe = _period(x.date, grid)
        st.append(dict(statement_id=f"{acct}_{pe}", account_id=acct, custodian="uploaded values", period_start=ps, period_end=pe,
                       ending_value=round(float(x.v), 2), source_file=name, notes=""))
        f = float(x.f) if not pd.isna(x.f) else 0.0
        if i == 0 and f == 0:
            f = float(x.v)          # first value is the opening deposit unless told otherwise
        if f != 0:
            fl.append(dict(account_id=acct, date=x.date.date(), amount=round(f, 2), flow_type="deposit" if f > 0 else "withdrawal",
                           description="from uploaded file", source_statement_id=f"{acct}_{pe}"))
    _write(out, st, fl, acct, label, "uploaded values and flows (nothing printed to reconcile against)")
    return dict(grid=grid, periods=len(d), first=str(d.date.iloc[0].date()), last=str(d.date.iloc[-1].date()), shape="values")


def from_template(files: dict[str, bytes], out: Path, label: str) -> dict:
    need = {"statements.csv"}
    got = {k.lower(): v for k, v in files.items()}
    if not need <= set(got):
        raise UploadError("the template upload needs statements.csv (flows.csv, positions.csv, accounts.csv optional)")
    out.mkdir(parents=True, exist_ok=True)
    for k in ["statements.csv", "flows.csv", "positions.csv", "accounts.csv"]:
        if k in got:
            if len(got[k]) > MAX_BYTES:
                raise UploadError(f"{k}: larger than 5 MB")
            (out / k).write_bytes(got[k])
    if "flows.csv" not in got:
        (out / "flows.csv").write_text("account_id,date,amount,flow_type,description,source_statement_id\n")
    if "positions.csv" not in got:
        (out / "positions.csv").write_text("statement_id,account_id,as_of_date,identifier,description,asset_class,quantity,price,market_value,weight_pct\n")
    st = pd.read_csv(out / "statements.csv")
    dates = pd.to_datetime(st.period_end, errors="coerce").dropna()
    grid = _grid(dates) if len(dates) > 1 else "A"
    return dict(grid=grid, periods=int(len(st)), first=str(dates.min().date()) if len(dates) else "", last=str(dates.max().date()) if len(dates) else "", shape="template")


def _write(out: Path, st: list, fl: list, acct: str, label: str, note: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(st).reindex(columns=_cols()).to_csv(out / "statements.csv", index=False)
    pd.DataFrame(fl, columns=["account_id", "date", "amount", "flow_type", "description", "source_statement_id"]).to_csv(out / "flows.csv", index=False)
    pd.DataFrame(columns=["statement_id", "account_id", "as_of_date", "identifier", "description", "asset_class", "quantity", "price", "market_value", "weight_pct"]).to_csv(out / "positions.csv", index=False)
    pd.DataFrame([dict(account_id=acct, label=label, owner_type="principal", discretionary="Y", strategy="default", benchmark="US_MKT", notes=note)]).to_csv(out / "accounts.csv", index=False)


TEMPLATES = {
    "returns.csv": "date,return\n2020-01-31,1.8\n2020-02-29,-3.1\n2020-03-31,-9.4\n",
    "values.csv": "date,value,flow\n2020-01-31,1000000,1000000\n2020-02-29,969000,0\n2020-03-31,905000,25000\n",
}
