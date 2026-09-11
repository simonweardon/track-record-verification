"""Synthetic dataset: ~10 accounts, staggered inception 1996-2013, annual
statements, dated deposits, year-end positions, with a known set of injected
defects so the reconciliation logic can be proven before real scans exist.

Nothing here is calibrated to the real record.  It exists only to exercise
the pipeline; delete it once real data is in.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .load import Dataset

TICKERS = [("AAPL", "us_equity"), ("MSFT", "us_equity"), ("JNJ", "us_equity"),
           ("BRK.B", "us_equity"), ("XOM", "us_equity"), ("PG", "us_equity"),
           ("JPM", "us_equity"), ("KO", "us_equity"), ("NSRGY", "intl_equity"),
           ("TM", "intl_equity"), ("NVS", "intl_equity"), ("UL", "intl_equity"),
           ("SNY", "intl_equity"), ("BP", "intl_equity"), ("SAP", "intl_equity")]
CUSTODIANS = ["Schwab", "Fidelity", "TDA"]
OPEN_YEARS = [1996, 1996, 1998, 2001, 2003, 2004, 2007, 2009, 2010, 2013, 2004, 2012]
CLOSE_YEARS = {10: 2009, 11: 2018}      # idx 10 capitulates Mar-2009; idx 11 closes normally
NON_DISCRETIONARY = {7}
PRINCIPAL = {0}
LAST_YEAR = 2025
TRUE_ALPHA = 0.02        # annual, injected on top of RF + beta*(MKT-RF)
TRUE_BETA = 1.0


@dataclass
class Manifest:
    """What was injected, so tests know what to expect."""
    expected_flags: list[dict] = field(default_factory=list)
    expected_unverified: list[str] = field(default_factory=list)
    true_returns: dict[str, float] = field(default_factory=dict)  # statement_id -> r
    true_alpha: float = TRUE_ALPHA
    true_beta: float = TRUE_BETA
    market_source: str = "random"          # "DEV_MKT" when built from real reference data


def _dietz_forward(begin, r, flows, p_start, p_end):
    cd = (p_end - p_start).days + 1
    end = begin * (1 + r)
    for d, amt in flows:
        w = (p_end - d).days / cd
        end += amt * (1 + r * w)
    return end


def _annual_market(years, rng):
    """Real developed-market annual returns and RF from the Ken French cache if
    present, else a random stand-in.  Returns (mkt, rf, source)."""
    try:
        from .reference import load_all
        ref = load_all()
        mkt, rf = {}, {}
        for y in years:
            m = ref.loc[f"{y}-01-01":f"{y}-12-31"]
            if len(m) != 12 or m["DEV_MKT"].isna().any():
                raise ValueError(f"reference data incomplete for {y}")
            mkt[y] = float((1 + m["DEV_MKT"]).prod() - 1)
            rf[y] = float((1 + m["DEV_RF"]).prod() - 1)
        return mkt, rf, "DEV_MKT"
    except Exception:
        return {y: rng.normal(0.08, 0.16) for y in years}, {y: 0.03 for y in years}, "random"


def generate(seed: int = 7, n_accounts: int = 12, defects: bool = True):
    rng = np.random.default_rng(seed)
    n_accounts = min(n_accounts, len(OPEN_YEARS))
    years = list(range(min(OPEN_YEARS[:n_accounts]), LAST_YEAR + 1))
    market, rf, source = _annual_market(years, rng)
    # shared idiosyncratic component, demeaned so the realized alpha over the sample
    # equals TRUE_ALPHA (a deterministic fixture for Phase 4 tests) while each year
    # still carries realistic noise around the market
    draws = rng.normal(0.0, 0.05, size=len(years))
    manager = dict(zip(years, draws - draws.mean()))

    st_rows, fl_rows, po_rows, ac_rows = [], [], [], []
    man = Manifest(market_source=source)

    for k in range(n_accounts):
        cust = CUSTODIANS[k % len(CUSTODIANS)]
        acct = f"{cust.upper()[:4]}_{1001 + k}"
        open_year, close_year = OPEN_YEARS[k], CLOSE_YEARS.get(k)
        beta = TRUE_BETA + float(rng.normal(0, 0.05))
        ac_rows.append(dict(account_id=acct,
                            label=("Principal" if k in PRINCIPAL else f"Client {k}"),
                            owner_type=("principal" if k in PRINCIPAL else "client"),
                            discretionary=("N" if k in NON_DISCRETIONARY else "Y"),
                            strategy="global_equity", benchmark="DEV_MKT",
                            notes=("client-directed; excluded from composite" if k in NON_DISCRETIONARY
                                   else ("closed" if close_year else ""))))
        prev_end = 0.0
        last_year = close_year or LAST_YEAR
        for y in range(open_year, last_year + 1):
            p_start, p_end = date(y, 1, 1), date(y, 12, 31)
            sid = f"{acct}_{y}"
            r = rf[y] + TRUE_ALPHA + beta * (market[y] - rf[y]) + manager[y] + float(rng.normal(0, 0.02))

            flows = []
            if y == open_year:
                d0 = date(y, int(rng.integers(1, 12)), int(rng.integers(1, 28)))
                flows.append((d0, round(float(rng.lognormal(12.5, 0.5)), 2), "deposit", "opening deposit"))
            n_dep = rng.poisson(1.2 if y - open_year < 6 else 0.4)
            for _ in range(n_dep):
                d = date(y, int(rng.integers(1, 13)), int(rng.integers(1, 29)))
                flows.append((d, round(float(rng.lognormal(10.8, 0.6)), 2), "deposit", "deposit"))
            if y - open_year > 8 and rng.random() < 0.12 and y != close_year:
                d = date(y, int(rng.integers(1, 13)), int(rng.integers(1, 29)))
                flows.append((d, -round(float(rng.lognormal(10.5, 0.5)), 2), "withdrawal", "withdrawal"))
            if y == close_year:   # full withdrawal so that ending value is exactly 0
                d = date(y, 3, 9) if k == 10 else date(y, int(rng.integers(6, 12)), 15)
                cd = (p_end - p_start).days + 1
                w = (p_end - d).days / cd
                interim = _dietz_forward(prev_end, r, [(dd, a) for dd, a, _, _ in flows], p_start, p_end)
                W = interim / (1 + r * w)
                flows.append((d, -round(W, 2), "withdrawal", "account closed - full withdrawal"))
            flows.sort()

            begin = prev_end
            end = round(_dietz_forward(begin, r, [(d, a) for d, a, _, _ in flows],
                                       p_start, p_end), 2)
            if y == close_year:
                end = 0.0
            dep = sum(a for _, a, _, _ in flows if a > 0)
            wdr = -sum(a for _, a, _, _ in flows if a < 0)
            pnl = round(end - begin - dep + wdr, 2)
            man.true_returns[sid] = r

            row = dict(statement_id=sid, account_id=acct, custodian=cust,
                       period_start=p_start, period_end=p_end, ending_value=end,
                       beginning_value=np.nan, stated_deposits=np.nan, stated_withdrawals=np.nan,
                       stated_income=np.nan, stated_fees=np.nan, stated_pnl=np.nan,
                       stated_return_pct=np.nan, source_file=f"{sid}.pdf", source_pages="1-6", notes="")
            # what each custodian "prints" — deliberately varied
            if cust == "Schwab":
                row.update(beginning_value=begin, stated_deposits=round(dep, 2),
                           stated_withdrawals=round(wdr, 2), stated_pnl=pnl)
            elif cust == "Fidelity":
                row.update(beginning_value=begin, stated_return_pct=round(r * 100, 2))
            # TDA: ending value only; verification must come from positions

            for d, a, t, desc in flows:
                fl_rows.append(dict(account_id=acct, date=d, amount=a, flow_type=t,
                                    description=desc, source_statement_id=sid))

            # positions: TDA account 5 has none before 2010 (simulates lost pages)
            if end == 0.0:
                if cust == "TDA":          # nothing printed, nothing held: unverifiable
                    man.expected_unverified.append(sid)
            elif not (k == 5 and y < 2010):
                n_pos = int(rng.integers(6, 13))
                idx = rng.choice(len(TICKERS), size=n_pos, replace=False)
                cash_w = rng.uniform(0.01, 0.05)
                w = rng.dirichlet(np.ones(n_pos)) * (1 - cash_w)
                mv = np.round(w * end, 2)
                for j, i in enumerate(idx):
                    tk, ac = TICKERS[i]
                    price = round(float(rng.uniform(20, 400)), 2)
                    po_rows.append(dict(statement_id=sid, account_id=acct, as_of_date=p_end,
                                        identifier=tk, description=f"{tk} common", asset_class=ac,
                                        quantity=round(mv[j] / price, 4), price=price,
                                        market_value=mv[j], weight_pct=round(100 * mv[j] / end, 2)))
                cash = round(end - float(mv.sum()), 2)
                po_rows.append(dict(statement_id=sid, account_id=acct, as_of_date=p_end,
                                    identifier="CASH", description="Cash & sweep", asset_class="cash",
                                    quantity=cash, price=1.0, market_value=cash,
                                    weight_pct=round(100 * cash / end, 2)))
            else:
                man.expected_unverified.append(sid)

            st_rows.append(row)
            prev_end = end

    st = pd.DataFrame(st_rows)
    fl = pd.DataFrame(fl_rows)
    po = pd.DataFrame(po_rows)
    ac = pd.DataFrame(ac_rows)

    if defects:
        _inject(st, fl, po, man)

    for c in ["period_start", "period_end"]:
        st[c] = pd.to_datetime(st[c])
    fl["date"] = pd.to_datetime(fl["date"])
    po["as_of_date"] = pd.to_datetime(po["as_of_date"])
    ds = Dataset(st, fl, po, [])
    ds.accounts = ac
    return ds, man


def _acct(k):
    return f"{CUSTODIANS[k % 3].upper()[:4]}_{1001 + k}"


def _append(df, row: dict):
    """df.loc[len(df)] silently overwrites after a drop; always use a fresh label."""
    df.loc[(df.index.max() + 1) if len(df) else 0] = row


def _inject(st, fl, po, man: Manifest):
    """Six realistic defects.  Each is what a real data-entry / OCR / custodian
    problem looks like, not a random perturbation."""
    E = man.expected_flags

    # 1. acct 0 (Schwab), 2003: OCR dropped the decimal point in ending value (x100)
    a, sid = _acct(0), f"{_acct(0)}_2003"
    i = st.index[st.statement_id == sid][0]
    st.at[i, "ending_value"] = round(st.at[i, "ending_value"] * 100, 2)
    E += [dict(statement_id=sid, code="IMPLAUSIBLE_RETURN"),
          dict(statement_id=sid, code="POSITIONS_SUM_MISMATCH"),
          dict(statement_id=sid, code="PNL_MISMATCH"),
          dict(statement_id=f"{a}_2004", code="CHAIN_BREAK")]

    # 2. acct 1 (Fidelity), 2012: statement's printed return disagrees with ours
    sid = f"{_acct(1)}_2012"
    i = st.index[st.statement_id == sid][0]
    st.at[i, "stated_return_pct"] = st.at[i, "stated_return_pct"] + 3.5
    E += [dict(statement_id=sid, code="RETURN_MISMATCH")]

    # 3. acct 2 (TDA), 2007: statement missing from binder entirely; a deposit
    #    that year is known from a bank record and was entered anyway
    a, sid = _acct(2), f"{_acct(2)}_2007"
    st.drop(st.index[st.statement_id == sid], inplace=True)
    po.drop(po.index[po.statement_id == sid], inplace=True)
    fl.drop(fl.index[fl.source_statement_id == sid], inplace=True)
    _append(fl, dict(account_id=a, date=date(2007, 6, 15), amount=40000.00,
                     flow_type="deposit", description="deposit (bank record)",
                     source_statement_id=""))
    E += [dict(statement_id=f"{a}_2008", code="PERIOD_GAP"),
          dict(statement_id=f"{a}_2008", code="NO_BEGINNING_VALUE"),
          dict(statement_id=None, account_id=a, code="ORPHAN_FLOW")]

    # 4. acct 3 (Schwab), 2019: one deposit row never entered
    a, sid = _acct(3), f"{_acct(3)}_2019"
    cand = fl.index[(fl.source_statement_id == sid) & (fl.amount > 0)]
    if len(cand) == 0:   # guarantee a deposit exists (keeping the statement internally
        # consistent: more deposits => less investment gain for the same ending value), then drop it
        _append(fl, dict(account_id=a, date=date(2019, 3, 3), amount=15000.0,
                         flow_type="deposit", description="deposit", source_statement_id=sid))
        i = st.index[st.statement_id == sid][0]
        st.at[i, "stated_deposits"] += 15000.0
        st.at[i, "stated_pnl"] -= 15000.0
        cand = fl.index[(fl.source_statement_id == sid) & (fl.amount > 0)]
    fl.drop(cand[0], inplace=True)
    E += [dict(statement_id=sid, code="FLOWS_MISMATCH"),
          dict(statement_id=sid, code="PNL_MISMATCH")]

    # 5. acct 4 (Fidelity), 2008: one position's market value keyed with an extra zero
    sid = f"{_acct(4)}_2008"
    j = po.index[(po.statement_id == sid) & (po.identifier != "CASH")][0]
    po.at[j, "market_value"] = round(po.at[j, "market_value"] * 10, 2)
    E += [dict(statement_id=sid, code="POSITIONS_SUM_MISMATCH")]

    # 6. acct 6 (Schwab), 2015: printed beginning != prior ending (an unrecorded
    #    in-kind transfer).  Statement is internally consistent, just doesn't chain.
    a, sid = _acct(6), f"{_acct(6)}_2015"
    i = st.index[st.statement_id == sid][0]
    bump = 25000.00
    st.at[i, "beginning_value"] += bump
    st.at[i, "ending_value"] = round(st.at[i, "ending_value"] + bump * 1.06, 2)
    st.at[i, "stated_pnl"] = round(st.at[i, "stated_pnl"] + bump * 0.06, 2)
    # keep positions consistent with the new ending value
    c = po.index[(po.statement_id == sid) & (po.identifier == "CASH")][0]
    po.at[c, "market_value"] = round(po.at[c, "market_value"] + bump * 1.06, 2)
    # and chain the following year so only 2015 breaks
    nxt = st.index[st.statement_id == f"{a}_2016"]
    if len(nxt):
        st.at[nxt[0], "beginning_value"] = st.at[i, "ending_value"]
        # propagate: recompute 2016+ endings from the new chain (Schwab prints pnl, keep it)
        for yy in range(2016, LAST_YEAR + 1):
            r_i = st.index[st.statement_id == f"{a}_{yy}"]
            if not len(r_i):
                break
            r_i = r_i[0]
            prev_i = st.index[st.statement_id == f"{a}_{yy-1}"][0]
            st.at[r_i, "beginning_value"] = st.at[prev_i, "ending_value"]
            new_end = round(st.at[r_i, "beginning_value"] + st.at[r_i, "stated_deposits"]
                            - st.at[r_i, "stated_withdrawals"] + st.at[r_i, "stated_pnl"], 2)
            delta = new_end - st.at[r_i, "ending_value"]
            st.at[r_i, "ending_value"] = new_end
            cc = po.index[(po.statement_id == f"{a}_{yy}") & (po.identifier == "CASH")][0]
            po.at[cc, "market_value"] = round(po.at[cc, "market_value"] + delta, 2)
    E += [dict(statement_id=sid, code="CHAIN_BREAK")]


def write(ds_and_manifest, out_dir: str | Path):
    ds, man = ds_and_manifest
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    st, fl, po = ds.statements.copy(), ds.flows.copy(), ds.positions.copy()
    for c in ["period_start", "period_end"]:
        st[c] = st[c].dt.strftime("%Y-%m-%d")
    fl["date"] = fl["date"].dt.strftime("%Y-%m-%d")
    po["as_of_date"] = po["as_of_date"].dt.strftime("%Y-%m-%d")
    st.to_csv(out / "statements.csv", index=False)
    fl.to_csv(out / "flows.csv", index=False)
    po.to_csv(out / "positions.csv", index=False)
    ds.accounts.to_csv(out / "accounts.csv", index=False)
    (out / "manifest.json").write_text(json.dumps(
        dict(expected_flags=man.expected_flags, expected_unverified=man.expected_unverified,
             true_returns=man.true_returns, true_alpha=man.true_alpha, true_beta=man.true_beta,
             market_source=man.market_source), indent=1))
