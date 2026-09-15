"""Cross-check the headline statistics against an independent R implementation.

`r/verify.R` recomputes, from `output/<fund>/phase4/aligned_data.csv` alone, the numbers
the Python pipeline reports in the same folder: CAPM / FF3 / Carhart4 / FF5 alpha with
Newey-West HAC standard errors and t-statistics, annualized return, volatility, Sharpe,
max drawdown, and the alpha-maxing score.  It shares no code with Python — the HAC
sandwich is written out in base R — so agreement is evidence the numbers are right and
not just reproducible.

This module runs that script over every built manager and writes the agreement to
`data/research/r-verify/`, which the research page renders.  Railway has no R, so the
CSVs are committed and the page is served from them; rebuild locally with

    python -m trackrecord r-verify

Anything above 1e-6 is treated as a disagreement worth reading, not a rounding artifact.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R_SCRIPT = ROOT / "r" / "verify.R"
OUT_DIR = ROOT / "data" / "research" / "r-verify"
FUNDS_OUT = ROOT / "output" / "funds"
FUNDS_DATA = ROOT / "data" / "funds"
TOL = 1e-6                      # agreement below this is floating-point noise, not a difference


def r_available() -> tuple[bool, str]:
    rs = shutil.which("Rscript")
    if not rs:
        return False, "Rscript not found"
    if not R_SCRIPT.exists():
        return False, "r/verify.R missing"
    p = subprocess.run([rs, "-e", 'cat(as.character(packageVersion("data.table")))'],
                       capture_output=True, text=True)
    if p.returncode != 0:
        return False, "data.table not installed"
    return True, f"R with data.table {p.stdout.strip()}"


def verify_fund(p4: Path) -> dict:
    """Run r/verify.R on one phase4 folder; return the agreement between R and Python."""
    ok, reason = r_available()
    if not ok:
        return dict(available=False, reason=reason)
    if not (p4 / "aligned_data.csv").exists():
        return dict(available=False, reason=f"{p4}/aligned_data.csv missing")
    try:
        p = subprocess.run([shutil.which("Rscript"), str(R_SCRIPT), str(p4)],
                           capture_output=True, text=True, timeout=300)
    except Exception as e:                                  # noqa: BLE001 - reported, not raised
        return dict(available=False, reason=str(e)[:200])
    if p.returncode != 0:
        return dict(available=False, reason=(p.stderr or p.stdout)[-400:])
    q = pd.read_csv(p4 / "verify_r.csv")
    worst = q.loc[q.abs_diff.idxmax()]
    return dict(available=True, quantities=q, n_quantities=len(q),
                max_abs_diff=float(q.abs_diff.max()), max_rel_diff=float(q.rel_diff.max()),
                worst_quantity=f"{worst.quantity} ({worst.model})" if worst.model != "-" else str(worst.quantity),
                agree=bool(q.abs_diff.max() < TOL), r_stdout=p.stdout.strip()[-300:])


def _pick(q: pd.DataFrame, quantity: str, model: str = "-") -> tuple[float, float]:
    r = q[(q.quantity == quantity) & (q.model == model)]
    if r.empty:
        return float("nan"), float("nan")
    return float(r.iloc[0].r_value), float(r.iloc[0].py_value)


def build(out_dir: Path | None = None, log=print) -> dict:
    """Run the R reproduction over every built manager; write the summary CSVs."""
    out = out_dir or OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    ok, reason = r_available()
    log(f"R reproduction: {reason}")
    rows, allq = [], []
    p4s = sorted(FUNDS_OUT.glob("*/phase4")) if FUNDS_OUT.exists() else []
    for p4 in p4s:
        slug = p4.parent.name
        meta_f = FUNDS_DATA / slug / "meta.json"
        meta = json.loads(meta_f.read_text()) if meta_f.exists() else {}
        res = verify_fund(p4)
        if not res.get("available"):
            log(f"  {slug}: skipped — {res.get('reason', '')[:120]}")
            continue
        q = res["quantities"]
        al = pd.read_csv(p4 / "aligned_data.csv")
        regs = pd.read_csv(p4 / "regressions.csv")
        a_r, a_py = _pick(q, "alpha_annual", "FF3")
        t_r, t_py = _pick(q, "t_alpha", "FF3")
        s_r, s_py = _pick(q, "alpha-maxing score")
        rows.append(dict(slug=slug, name=meta.get("name", slug), manager=meta.get("manager", ""),
                         cells=len(al), factor_set=str(regs.factor_set.iloc[0]),
                         quantities=res["n_quantities"], max_abs_diff=res["max_abs_diff"],
                         max_rel_diff=res["max_rel_diff"], agree=res["agree"],
                         worst_quantity=res["worst_quantity"],
                         ff3_alpha_r=a_r, ff3_alpha_python=a_py, ff3_t_r=t_r, ff3_t_python=t_py,
                         score_r=s_r, score_python=s_py))
        q = q.copy(); q.insert(0, "slug", slug)
        allq.append(q)
        log(f"  {slug}: {res['n_quantities']} quantities, max |R − Python| {res['max_abs_diff']:.2e}")
    summary = pd.DataFrame(rows)
    quantities = pd.concat(allq, ignore_index=True) if allq else pd.DataFrame()
    by_q = pd.DataFrame()
    if not quantities.empty:
        by_q = (quantities.groupby(["quantity", "model"])
                .agg(managers=("abs_diff", "size"), max_abs_diff=("abs_diff", "max"),
                     max_rel_diff=("rel_diff", "max"))
                .reset_index().sort_values("max_abs_diff", ascending=False))
    summary.to_csv(out / "summary.csv", index=False)
    by_q.to_csv(out / "by_quantity.csv", index=False)
    man = dict(built=datetime.now().strftime("%Y-%m-%d %H:%M"), r=reason, available=ok,
               managers=len(summary), quantities_per_manager=int(summary.quantities.max()) if len(summary) else 0,
               checks=int(len(quantities)), tol=TOL,
               max_abs_diff=float(summary.max_abs_diff.max()) if len(summary) else float("nan"),
               disagreements=int((~summary.agree).sum()) if len(summary) else 0)
    (out / "manifest.json").write_text(json.dumps(man, indent=2))
    log(f"  wrote {len(summary)} managers, {man['checks']} checks, max |R − Python| "
        f"{man['max_abs_diff']:.2e} to {out}")
    return man
