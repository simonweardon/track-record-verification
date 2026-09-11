"""End-to-end: synth -> report, through the CLI, with small simulation counts."""
from pathlib import Path
import pytest

from trackrecord.cli import main


def test_full_pipeline_via_cli(tmp_path):
    try:
        from trackrecord.reference import load_all
        load_all()
    except Exception as e:
        pytest.skip(f"reference data unavailable: {e}")
    data = tmp_path / "synthetic"
    assert main(["synth", "--out", str(data)]) == 0
    out = tmp_path / "out"
    assert main(["report", "--data", str(data), "--out", str(out), "--claimed", "0.14",
                 "--n-boot", "200", "--n-cohort", "300"]) == 0
    for f in ["REPORT.md", "dashboard.html", "reconciliation.csv", "coverage.md", "flags.csv", "statements_reconciled.csv",
              "phase2/summary.md", "phase2/composite_returns.csv", "phase2/account_summary.csv",
              "phase3/summary.md", "phase3/allocation.csv",
              "phase4/summary.md", "phase4/regressions.csv", "phase4/cohort.csv", "phase4/rolling.csv"]:
        assert (out / f).exists(), f
    t = (out / "REPORT.md").read_text()
    assert "PLACEHOLDER DATA" in t
    assert "Claimed: **+14.00%**" in t
    assert "### Verified" in t and "### Estimated" in t and "### Unknown" in t


def test_reconcile_exit_code_reflects_flags(tmp_path):
    data = tmp_path / "d"
    main(["synth", "--out", str(data), "--clean"])
    assert main(["reconcile", "--data", str(data), "--out", str(tmp_path / "o1")]) == 0
    main(["synth", "--out", str(data)])
    assert main(["reconcile", "--data", str(data), "--out", str(tmp_path / "o2")]) == 1
