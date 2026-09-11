"""Public reference data: Ken French factor library (monthly, percent → decimal).

Used for benchmarks (Mkt-RF + RF = total market return) in Phase 2 and for
factor regressions in Phase 4.  Files are fetched once and cached as clean
CSVs in data/reference/.  These are real data, not placeholders.

Series ids available to benchmarks / regressions after load_all():
  US:     US_MKT_RF, US_SMB, US_HML, US_RF, US_RMW, US_CMA, US_MOM,  US_MKT (=Mkt-RF+RF)
  Devel.: DEV_MKT_RF, DEV_SMB, DEV_HML, DEV_RF, DEV_RMW, DEV_CMA, DEV_MOM, DEV_MKT
  Dev ex-US: DXUS_MKT_RF, DXUS_SMB, DXUS_HML, DXUS_RF, DXUS_MKT
"""
from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FILES = {  # key: (zip, prefix for column names)
    "us3":     ("F-F_Research_Data_Factors_CSV.zip", "US"),
    "us5":     ("F-F_Research_Data_5_Factors_2x3_CSV.zip", "US"),
    "usmom":   ("F-F_Momentum_Factor_CSV.zip", "US"),
    "dev3":    ("Developed_3_Factors_CSV.zip", "DEV"),
    "dev5":    ("Developed_5_Factors_CSV.zip", "DEV"),
    "devmom":  ("Developed_Mom_Factor_CSV.zip", "DEV"),
    "dxus3":   ("Developed_ex_US_3_Factors_CSV.zip", "DXUS"),
}
DEFAULT_DIR = Path("data/reference")


def _parse_monthly(text: str) -> pd.DataFrame:
    """French CSVs: preamble, a header line starting with ',', monthly rows
    keyed YYYYMM, then a blank line and an annual section we ignore."""
    lines = text.splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith(","))
    header = [h.strip() for h in lines[start].split(",")]
    rows = []
    for l in lines[start + 1:]:
        cells = [c.strip() for c in l.split(",")]
        if len(cells) < 2 or not cells[0].isdigit() or len(cells[0]) != 6:
            break
        rows.append(cells)
    df = pd.DataFrame(rows, columns=header)
    df = df.rename(columns={header[0]: "ym"})
    df.index = pd.to_datetime(df.pop("ym"), format="%Y%m") + pd.offsets.MonthEnd(0)
    df.index.name = "month"
    df = df.apply(pd.to_numeric)
    df = df.replace(-99.99, pd.NA).astype(float) / 100.0
    df.columns = [c.replace("-", "_").upper() for c in df.columns]
    return df


def _parse_annual(text: str) -> pd.DataFrame:
    """The 'Annual Factors: January-December' block: rows keyed YYYY."""
    lines = text.splitlines()
    try:
        a = next(i for i, l in enumerate(lines) if l.strip().lower().startswith("annual factors"))
    except StopIteration:
        return pd.DataFrame()
    start = next(i for i in range(a, len(lines)) if lines[i].startswith(","))
    header = [h.strip() for h in lines[start].split(",")]
    rows = []
    for l in lines[start + 1:]:
        cells = [c.strip() for c in l.split(",")]
        if len(cells) < 2 or not cells[0].isdigit() or len(cells[0]) != 4:
            break
        rows.append(cells)
    df = pd.DataFrame(rows, columns=header).rename(columns={header[0]: "year"})
    df.index = df.pop("year").astype(int)
    df.index.name = "year"
    df = df.apply(pd.to_numeric).replace(-99.99, pd.NA).astype(float) / 100.0
    df.columns = [c.replace("-", "_").upper() for c in df.columns]
    return df


def _text(key: str, ref_dir: Path, force: bool) -> str:
    zip_name, _ = FILES[key]
    zpath = ref_dir / zip_name
    if not zpath.exists() or force:
        urllib.request.urlretrieve(BASE + zip_name, zpath)
    with zipfile.ZipFile(zpath) as z:
        name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
        return z.read(name).decode("utf-8", errors="replace")


def fetch(key: str, ref_dir: Path = DEFAULT_DIR, force: bool = False, freq: str = "M") -> pd.DataFrame:
    _, prefix = FILES[key]
    ref_dir.mkdir(parents=True, exist_ok=True)
    cache = ref_dir / f"{key}_{'monthly' if freq == 'M' else 'annual'}.csv"
    if cache.exists() and not force:
        return pd.read_csv(cache, index_col=0, parse_dates=(freq == "M"))
    text = _text(key, ref_dir, force)
    df = _parse_monthly(text) if freq == "M" else _parse_annual(text)
    df.columns = [f"{prefix}_{c}" for c in df.columns]
    df.to_csv(cache)
    return df


def load_all(ref_dir: Path = DEFAULT_DIR, force: bool = False, freq: str = "M") -> pd.DataFrame:
    """One wide frame (monthly by default, annual with freq='A') of every factor
    plus total-market series.  Developed momentum is published as WML; aliased to MOM."""
    frames = []
    for key in FILES:
        try:
            frames.append(fetch(key, ref_dir, force, freq))
        except Exception as e:   # a single missing file shouldn't kill the run
            print(f"warning: could not load {key}: {e}")
    wide = pd.concat(frames, axis=1)
    wide = wide.loc[:, ~wide.columns.duplicated()]
    for p in ["US", "DEV", "DXUS"]:
        if f"{p}_MKT_RF" in wide and f"{p}_RF" in wide:
            wide[f"{p}_MKT"] = wide[f"{p}_MKT_RF"] + wide[f"{p}_RF"]
        if f"{p}_WML" in wide and f"{p}_MOM" not in wide:
            wide[f"{p}_MOM"] = wide[f"{p}_WML"]
    return wide.sort_index()


def benchmark_series(ref: pd.DataFrame, spec: str) -> pd.Series:
    """'US_MKT'  or a monthly-rebalanced blend  '60:US_MKT,40:DXUS_MKT'."""
    if ":" not in spec:
        return ref[spec].rename(spec)
    parts = [p.split(":") for p in spec.split(",")]
    w = {s.strip(): float(x) for x, s in parts}
    tot = sum(w.values())
    out = sum(ref[s] * (x / tot) for s, x in w.items())
    return out.rename(spec)


def compound_to_periods(monthly: pd.Series, periods: pd.DataFrame) -> pd.Series:
    """Geometrically link a monthly series over each [period_start, period_end].
    Periods must align to whole months; NaN where months are missing."""
    out = []
    for _, p in periods.iterrows():
        m = monthly[(monthly.index >= p.period_start) & (monthly.index <= p.period_end)]
        n_expected = (p.period_end.year - p.period_start.year) * 12 + p.period_end.month - p.period_start.month + 1
        if len(m) != n_expected or m.isna().any():
            out.append(float("nan"))
        else:
            out.append(float((1 + m).prod() - 1))
    return pd.Series(out, index=periods.index)
