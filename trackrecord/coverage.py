"""Coverage report: which account-years have verified data, which are flagged,
which are unverified, and where the holes are.  No interpolation."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .reconcile import ReconcileResult, ERROR, WARN

SYMBOL = {"verified": "V", "flagged": "X", "unverified": "U"}
RANK = {"flagged": 0, "unverified": 1, "verified": 2}   # worst first


def year_matrix(st: pd.DataFrame) -> pd.DataFrame:
    """Rows = accounts, cols = calendar years, cell = worst status among
    statements whose period_end falls in that year; '-' = hole inside the
    account's observed span; '' = outside span."""
    st = st.dropna(subset=["period_end"])
    if st.empty:
        return pd.DataFrame()
    st = st.assign(year=st["period_end"].dt.year)
    y0, y1 = int(st["year"].min()), int(st["year"].max())
    years = list(range(y0, y1 + 1))
    rows = {}
    for acct, g in st.groupby("account_id"):
        a0, a1 = int(g["year"].min()), int(g["year"].max())
        row = {}
        for y in years:
            if y < a0 or y > a1:
                row[y] = ""
            else:
                sub = g[g["year"] == y]
                if sub.empty:
                    row[y] = "-"
                else:
                    worst = min(sub["status"], key=lambda s: RANK[s])
                    row[y] = SYMBOL[worst] + (f"{len(sub)}" if len(sub) > 1 else "")
        rows[acct] = row
    return pd.DataFrame(rows).T[years]


def gaps(st: pd.DataFrame) -> pd.DataFrame:
    out = []
    for acct, g in st.dropna(subset=["period_start", "period_end"]).groupby("account_id"):
        g = g.sort_values("period_end")
        prev_end = None
        for _, s in g.iterrows():
            if prev_end is not None and (s.period_start - prev_end).days > 1:
                out.append({"account_id": acct,
                            "gap_start": (prev_end + pd.Timedelta(days=1)).date(),
                            "gap_end": (s.period_start - pd.Timedelta(days=1)).date(),
                            "days": (s.period_start - prev_end).days - 1})
            prev_end = s.period_end
    return pd.DataFrame(out, columns=["account_id", "gap_start", "gap_end", "days"])


def account_summary(st: pd.DataFrame, gp: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for acct, g in st.dropna(subset=["period_end"]).groupby("account_id"):
        span_days = (g["period_end"].max() - g["period_start"].min()).days + 1
        covered = int(((g["period_end"] - g["period_start"]).dt.days + 1).sum())
        ag = gp[gp["account_id"] == acct]
        rows.append({
            "account_id": acct,
            "custodian": ";".join(sorted(g["custodian"].dropna().unique())),
            "first_period_start": g["period_start"].min().date(),
            "last_period_end": g["period_end"].max().date(),
            "statements": len(g),
            "verified": int((g["status"] == "verified").sum()),
            "flagged": int((g["status"] == "flagged").sum()),
            "unverified": int((g["status"] == "unverified").sum()),
            "gaps": len(ag),
            "days_missing": int(ag["days"].sum()) if len(ag) else 0,
            "pct_of_span_covered": round(100 * covered / span_days, 1) if span_days else 0,
            "with_positions": int((g["n_positions"] > 0).sum()),
            "returns_computable": int(g["dietz_return"].notna().sum()),
        })
    return pd.DataFrame(rows)


def write_report(res: ReconcileResult, out_dir: str | Path, title: str = "Coverage report") -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    st, fl = res.statements, res.flags
    mat = year_matrix(st)
    gp = gaps(st)
    acc = account_summary(st, gp)
    summ = res.summary()

    st_out = st.copy()
    for c in ["period_start", "period_end"]:
        st_out[c] = st_out[c].dt.date
    st_out.to_csv(out / "statements_reconciled.csv", index=False)
    fl.to_csv(out / "flags.csv", index=False)
    mat.to_csv(out / "coverage_matrix.csv")
    acc.to_csv(out / "account_summary.csv", index=False)
    gp.to_csv(out / "gaps.csv", index=False)

    L: list[str] = [f"# {title}", ""]
    L += ["## Totals", ""]
    L += [f"- Accounts: **{summ['accounts']}**",
          f"- Statement periods: **{summ['statements']}** — "
          f"verified {summ['verified']}, flagged {summ['flagged']}, unverified {summ['unverified']}",
          f"- Flags: {summ['errors']} errors, {summ['warnings']} warnings",
          f"- Load problems (rows the loader could not use): {summ['load_problems']}", ""]
    L += ["Status meanings: **V** verified (>=1 independent check passed, none failed) · "
          "**X** flagged (a check failed — see flags) · **U** unverified (a value exists but "
          "nothing on the statement confirms it) · **-** no statement for this year inside the "
          "account's observed span · blank = outside observed span. A digit after the symbol "
          "means several statements ended in that year.", ""]

    L += ["## Account × year", ""]
    if not mat.empty:
        cols = [str(c) for c in mat.columns]
        L.append("| account | " + " | ".join(cols) + " |")
        L.append("|---|" + "|".join(["---"] * len(cols)) + "|")
        for acct, row in mat.iterrows():
            L.append(f"| {acct} | " + " | ".join(str(v) if v != "" else " " for v in row) + " |")
    L.append("")

    L += ["## Per account", ""]
    if len(acc):
        L.append("| " + " | ".join(acc.columns) + " |")
        L.append("|" + "|".join(["---"] * len(acc.columns)) + "|")
        for _, r in acc.iterrows():
            L.append("| " + " | ".join(str(v) for v in r.values) + " |")
    L.append("")

    L += ["## Gaps (no statement covers these dates)", ""]
    if len(gp):
        for _, g in gp.iterrows():
            L.append(f"- {g.account_id}: {g.gap_start} → {g.gap_end} ({g.days} days)")
    else:
        L.append("- none")
    L.append("")

    for sev, head in [(ERROR, "Errors (reconciliation failures — must be resolved or disclosed)"),
                      (WARN, "Warnings")]:
        sub = fl[fl["severity"] == sev] if len(fl) else fl
        L += [f"## {head}", ""]
        if len(sub):
            for _, f in sub.iterrows():
                sid = f.statement_id if pd.notna(f.statement_id) else f"(account {f.account_id})"
                L.append(f"- `{sid}` **{f.code}** — {f.detail}")
        else:
            L.append("- none")
        L.append("")

    if res.problems:
        L += ["## Load problems", ""] + [f"- {p}" for p in res.problems] + [""]

    path = out / "coverage.md"
    path.write_text("\n".join(L))
    return path
