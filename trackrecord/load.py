"""Read the three normalized CSVs, coerce types, validate, return a Dataset."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .schema import (ACCOUNT_COLUMNS, ASSET_CLASSES, FLOW_TYPES, TABLES)


@dataclass
class Dataset:
    statements: pd.DataFrame
    flows: pd.DataFrame
    positions: pd.DataFrame
    problems: list[str] = field(default_factory=list)  # row-level validation issues
    accounts: pd.DataFrame | None = None               # accounts.csv metadata (Phase 2)


class SchemaError(Exception):
    """Structural problem (missing file / missing column). Not recoverable."""


def _read(path: Path, name: str) -> pd.DataFrame:
    columns, required = TABLES[name]
    if not path.exists():
        if name == "statements":
            raise SchemaError(f"{path} is required")
        return pd.DataFrame({c: pd.Series(dtype="object") for c in columns})

    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SchemaError(f"{path}: missing required columns {missing}")
    for c in columns:
        if c not in df.columns:
            df[c] = ""

    # blank -> NA, then coerce
    df = df.replace({"": pd.NA})
    for c, kind in columns.items():
        if kind == "date":
            df[c] = pd.to_datetime(df[c], errors="coerce", format="%Y-%m-%d")
        elif kind == "float":
            df[c] = pd.to_numeric(
                df[c].astype("string").str.replace(",", "", regex=False)
                     .str.replace("$", "", regex=False),
                errors="coerce",
            )
        else:
            df[c] = df[c].astype("string").str.strip()
    return df[list(columns)]


def _check_required(df: pd.DataFrame, name: str, problems: list[str]) -> None:
    _, required = TABLES[name]
    for c in required:
        bad = df.index[df[c].isna()]
        for i in bad:
            problems.append(f"{name}.csv row {i + 2}: required column '{c}' is blank or unparseable")


def load(data_dir: str | Path) -> Dataset:
    data_dir = Path(data_dir)
    problems: list[str] = []

    st = _read(data_dir / "statements.csv", "statements")
    fl = _read(data_dir / "flows.csv", "flows")
    po = _read(data_dir / "positions.csv", "positions")

    _check_required(st, "statements", problems)
    _check_required(fl, "flows", problems)
    _check_required(po, "positions", problems)

    dup = st["statement_id"][st["statement_id"].duplicated(keep=False)].dropna().unique()
    for d in dup:
        problems.append(f"statements.csv: statement_id '{d}' appears more than once")

    bad_end = st.index[(st["period_end"] < st["period_start"])]
    for i in bad_end:
        problems.append(f"statements.csv row {i + 2}: period_end before period_start")

    bad_ft = fl.index[~fl["flow_type"].isin(FLOW_TYPES) & fl["flow_type"].notna()]
    for i in bad_ft:
        problems.append(f"flows.csv row {i + 2}: flow_type '{fl.at[i, 'flow_type']}' "
                        f"not in {sorted(FLOW_TYPES)}")

    bad_ac = po.index[~po["asset_class"].isin(ASSET_CLASSES) & po["asset_class"].notna()]
    for i in bad_ac:
        problems.append(f"positions.csv row {i + 2}: asset_class '{po.at[i, 'asset_class']}' "
                        f"not in {sorted(ASSET_CLASSES)}")

    known_st = set(st["statement_id"].dropna())
    orphan_pos = po.index[~po["statement_id"].isin(known_st) & po["statement_id"].notna()]
    for i in orphan_pos:
        problems.append(f"positions.csv row {i + 2}: statement_id '{po.at[i, 'statement_id']}' "
                        f"not in statements.csv")

    # sign sanity on flows: deposits/transfer_in should be +, withdrawals/transfer_out -
    for i, r in fl.dropna(subset=["amount", "flow_type"]).iterrows():
        if r.flow_type in ("deposit", "transfer_in") and r.amount < 0:
            problems.append(f"flows.csv row {i + 2}: {r.flow_type} has negative amount")
        if r.flow_type in ("withdrawal", "transfer_out") and r.amount > 0:
            problems.append(f"flows.csv row {i + 2}: {r.flow_type} has positive amount")

    st = st.sort_values(["account_id", "period_end"]).reset_index(drop=True)
    fl = fl.sort_values(["account_id", "date"]).reset_index(drop=True)
    po = po.sort_values(["account_id", "as_of_date", "identifier"]).reset_index(drop=True)
    ds = Dataset(st, fl, po, problems)
    acc_path = data_dir / "accounts.csv"
    if acc_path.exists():
        ac = pd.read_csv(acc_path, dtype=str, keep_default_na=False).replace({"": pd.NA})
        for c in ACCOUNT_COLUMNS:
            if c not in ac.columns:
                ac[c] = pd.NA
        unknown = set(ac["account_id"].dropna()) - set(st["account_id"].dropna())
        for u in sorted(unknown):
            problems.append(f"accounts.csv: account_id '{u}' has no statements")
        ds.accounts = ac[list(ACCOUNT_COLUMNS)]
    return ds
