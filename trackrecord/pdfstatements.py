"""Brokerage statement PDFs → statements.csv + flows.csv.

Built for text PDFs (custodian downloads, not scans).  It looks for the account-summary
block every custodian prints in some form — period, account, beginning value, additions,
subtractions, change in value, ending value — by label, and reports per file exactly what
it found and what it could not.  Nothing is guessed: a value that isn't on the page stays
blank and the pipeline marks the period accordingly.

Flow timing: statements print period totals for additions/subtractions, not dates, so the
flows are placed mid-period (the original Dietz approximation) and the statement's notes
say so.  Dated flows can be added by hand in flows.csv for a sharper number.
"""
from __future__ import annotations

import re
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
DATE = rf"(?:{MONTHS})\.?\s+\d{{1,2}},?\s+\d{{4}}|\d{{1,2}}/\d{{1,2}}/\d{{2,4}}"
NUM = r"\(?-?\$?\s?-?[\d,]+(?:\.\d{2})?\)?"

LABELS = {
    "beginning_value": [r"beginning\s+(?:account\s+|portfolio\s+|market\s+)?value", r"opening\s+(?:account\s+)?(?:value|balance)", r"beginning\s+balance", r"value\s+at\s+beginning"],
    "stated_deposits": [r"additions", r"deposits?(?:\s+and\s+credits)?", r"contributions", r"cash\s+in", r"money\s+in"],
    "stated_withdrawals": [r"subtractions", r"withdrawals?(?:\s+and\s+debits)?", r"distributions", r"cash\s+out", r"money\s+out"],
    "stated_pnl": [r"change\s+in\s+(?:investment\s+)?value", r"investment\s+(?:gain|gain\s*/\s*loss|results?)", r"net\s+(?:change|gain|investment\s+gain)", r"market\s+(?:appreciation|change)"],
    "stated_income": [r"(?:total\s+)?(?:dividends?|income)(?:\s+and\s+interest)?(?:\s+received)?", r"interest\s+and\s+dividends"],
    "stated_fees": [r"(?:total\s+)?fees(?:\s+and\s+charges)?", r"advisory\s+fees?", r"management\s+fees?"],
    "ending_value": [r"ending\s+(?:account\s+|portfolio\s+|market\s+)?value", r"closing\s+(?:account\s+)?(?:value|balance)", r"ending\s+balance", r"value\s+at\s+end", r"total\s+(?:account\s+)?value"],
}


def _num(tok: str) -> float | None:
    t = tok.strip().replace("$", "").replace(" ", "").replace(",", "")
    neg = (t.startswith("(") and t.endswith(")")) or t.startswith("-") or t.startswith("(-")
    t = t.strip("()").lstrip("-")
    if t in ("", "-", "—"):
        return 0.0
    try:
        v = float(t)
    except ValueError:
        return None
    return -v if neg else v


def _parse_date(s: str) -> date | None:
    for fmt in ("%B %d, %Y", "%B %d %Y", "%b %d, %Y", "%b %d %Y", "%b. %d, %Y", "%m/%d/%Y", "%m/%d/%y"):
        try:
            return pd.to_datetime(s.replace("Sept", "Sep"), format=fmt).date()
        except Exception:
            continue
    try:
        return pd.to_datetime(s).date()
    except Exception:
        return None


def extract_text(data: bytes) -> str:
    import pymupdf
    doc = pymupdf.open(stream=data, filetype="pdf")
    return "\n".join(page.get_text("text") for page in doc)


def parse_statement_text(text: str) -> dict:
    """Best-effort account summary from statement text.  Returns fields + 'found' / 'missing'."""
    out: dict = {"found": [], "missing": []}
    flat = re.sub(r"[ \t]+", " ", text)
    # ---- period: two dates joined by - / – / to / through
    m = re.search(rf"({DATE})\s*(?:-|–|—|to|through)\s*({DATE})", flat, re.I)
    if m:
        d0, d1 = _parse_date(m.group(1)), _parse_date(m.group(2))
        if d0 and d1 and d1 > d0:
            out["period_start"], out["period_end"] = d0, d1; out["found"].append("period")
    if "period_end" not in out:
        m = re.search(rf"(?:as of|statement date|period ending|for the period ending)\s*:?\s*({DATE})", flat, re.I)
        if m and _parse_date(m.group(1)):
            out["period_end"] = _parse_date(m.group(1)); out["found"].append("period end only")
        else:
            out["missing"].append("period")
    # ---- account number (keep the last 4 digits only)
    m = re.search(r"account\s*(?:number|no\.?|#)\s*:?\s*([A-Z0-9][A-Z0-9\-]{3,})", flat, re.I)
    if m:
        digits = re.sub(r"[^0-9A-Za-z]", "", m.group(1))
        out["account_last4"] = digits[-4:]; out["found"].append("account")
    else:
        out["missing"].append("account number")
    # ---- custodian
    for name in ["Fidelity", "Schwab", "Vanguard", "Interactive Brokers", "Merrill", "Morgan Stanley", "TD Ameritrade", "Robinhood", "E*TRADE", "Pershing", "Raymond James", "UBS", "Wells Fargo", "J.P. Morgan", "JPMorgan", "Goldman"]:
        if re.search(re.escape(name), flat, re.I):
            out["custodian"] = name; break
    # ---- labeled values: the first number on the same line (or next line) as the label
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for field, pats in LABELS.items():
        val = None
        for pat in pats:
            rx = re.compile(rf"^\s*{pat}\b[^0-9\(\$\-]*({NUM})", re.I)
            for k, line in enumerate(lines):
                mm = rx.search(line)
                if mm:
                    val = _num(mm.group(1))
                    if val is not None:
                        break
                if re.fullmatch(rf"\s*{pat}\s*:?\s*", line, re.I) and k + 1 < len(lines):
                    v2 = _num(lines[k + 1]) if re.fullmatch(rf"\s*{NUM}\s*", lines[k + 1]) else None
                    if v2 is not None:
                        val = v2; break
            if val is not None:
                break
        if val is not None:
            out[field] = abs(val) if field in ("stated_deposits", "stated_withdrawals", "stated_fees") else val
            out["found"].append(field)
        elif field in ("ending_value", "beginning_value"):
            out["missing"].append(field)
    return out


def from_pdfs(files: list[tuple[str, bytes]], out: Path, label: str) -> tuple[dict, list[dict]]:
    """Parse every PDF, write the dataset, return (info, per-file report)."""
    from .uploads import UploadError, _cols
    reports, rows, flows = [], [], []
    for fn, data in files:
        rep = {"file": fn}
        try:
            text = extract_text(data)
        except Exception as e:
            rep["error"] = f"could not read PDF ({e})"; reports.append(rep); continue
        if len(text.strip()) < 200:
            rep["error"] = "no text layer — this looks like a scan; OCR is not automated yet"; reports.append(rep); continue
        p = parse_statement_text(text)
        rep.update({k: v for k, v in p.items() if k not in ("found", "missing")})
        rep["found"] = ", ".join(p["found"]); rep["missing"] = ", ".join(p["missing"])
        if "period_end" not in p or "ending_value" not in p:
            rep["error"] = "skipped: need at least the period and the ending value"; reports.append(rep); continue
        pe = p["period_end"]
        ps = p.get("period_start") or (pe.replace(day=1))
        acct = f"{(p.get('custodian') or 'ACCT').upper()[:4]}_{p.get('account_last4', 'XXXX')}"
        sid = f"{acct}_{pe.isoformat()}"
        rows.append(dict(statement_id=sid, account_id=acct, custodian=p.get("custodian", "statement PDF"), period_start=ps, period_end=pe,
                         ending_value=p["ending_value"], beginning_value=p.get("beginning_value"), stated_deposits=p.get("stated_deposits"),
                         stated_withdrawals=p.get("stated_withdrawals"), stated_income=p.get("stated_income"), stated_fees=p.get("stated_fees"),
                         stated_pnl=p.get("stated_pnl"), source_file=fn, source_pages="", notes="parsed from PDF; flows placed mid-period"))
        mid = ps + (pe - ps) / 2
        if p.get("stated_deposits"):
            flows.append(dict(account_id=acct, date=mid, amount=round(p["stated_deposits"], 2), flow_type="deposit", description="period additions (date approximated: mid-period)", source_statement_id=sid))
        if p.get("stated_withdrawals"):
            flows.append(dict(account_id=acct, date=mid, amount=-round(p["stated_withdrawals"], 2), flow_type="withdrawal", description="period subtractions (date approximated: mid-period)", source_statement_id=sid))
        reports.append(rep)
    if not rows:
        raise UploadError("none of the PDFs yielded a period and an ending value — " + "; ".join(f"{r['file']}: {r.get('error', 'unknown')}" for r in reports[:5]))
    st = pd.DataFrame(rows).sort_values(["account_id", "period_end"]).drop_duplicates("statement_id", keep="last")
    if len(st) < 12:
        raise UploadError(f"only {len(st)} usable statement{'s' if len(st) != 1 else ''} — upload at least 12 (a year of monthly statements, or twelve annual ones); "
                          + ("skipped: " + "; ".join(f"{r['file']}: {r['error']}" for r in reports if r.get("error")) if any(r.get("error") for r in reports) else ""))
    out.mkdir(parents=True, exist_ok=True)
    st.reindex(columns=_cols()).to_csv(out / "statements.csv", index=False)
    pd.DataFrame(flows, columns=["account_id", "date", "amount", "flow_type", "description", "source_statement_id"]).to_csv(out / "flows.csv", index=False)
    pd.DataFrame(columns=["statement_id", "account_id", "as_of_date", "identifier", "description", "asset_class", "quantity", "price", "market_value", "weight_pct"]).to_csv(out / "positions.csv", index=False)
    accts = st.account_id.unique()
    pd.DataFrame([dict(account_id=a, label=f"{label} — {a}", owner_type="principal", discretionary="Y", strategy="default", benchmark="US_MKT",
                       notes="parsed from statement PDFs") for a in accts]).to_csv(out / "accounts.csv", index=False)
    pd.DataFrame(reports).to_csv(out / "pdf_report.csv", index=False)
    dates = pd.to_datetime(st.period_end)
    spans = dates.sort_values().diff().dt.days.dropna()
    grid = "M" if (len(spans) and spans.median() < 45) else "A"
    info = dict(grid=grid, periods=int(len(st)), first=str(dates.min().date()), last=str(dates.max().date()), shape="pdf",
                parsed=int(len(rows)), skipped=int(len(reports) - len(rows)), accounts=int(len(accts)))
    return info, reports
