"""The synthetic dataset injects known defects; the reconciler must find all of
them and nothing else at error severity."""
import numpy as np
import pytest

from trackrecord.synthetic import generate
from trackrecord.reconcile import reconcile, ERROR, WARN


@pytest.fixture(scope="module")
def dirty():
    ds, man = generate(seed=7, defects=True)
    return reconcile(ds), man


@pytest.fixture(scope="module")
def clean():
    ds, man = generate(seed=7, defects=False)
    return reconcile(ds), man


def test_every_injected_defect_is_found(dirty):
    res, man = dirty
    found = {(r.statement_id if isinstance(r.statement_id, str) else None, r.account_id, r.code)
             for r in res.flags.itertuples()}
    for e in man.expected_flags:
        sid = e.get("statement_id")
        if sid is None:
            assert any(f[0] is None and f[1] == e["account_id"] and f[2] == e["code"] for f in found), e
        else:
            assert any(f[0] == sid and f[2] == e["code"] for f in found), e


def test_no_unexpected_errors(dirty):
    res, man = dirty
    expected = {(e.get("statement_id"), e["code"]) for e in man.expected_flags}
    errs = res.flags[res.flags.severity == ERROR]
    unexpected = [(r.statement_id if isinstance(r.statement_id, str) else None, r.code)
                  for r in errs.itertuples()
                  if (r.statement_id if isinstance(r.statement_id, str) else None, r.code) not in expected]
    assert unexpected == []


def test_unverified_are_exactly_the_positionless_ending_only_statements(dirty):
    res, man = dirty
    st = res.statements.set_index("statement_id")
    assert set(st.index[st.status == "unverified"]) == set(man.expected_unverified)


def test_clean_dataset_has_no_errors_or_warnings(clean):
    res, _ = clean
    assert res.summary()["errors"] == 0
    assert res.summary()["flagged"] == 0
    assert (res.flags.severity != WARN).all()


def test_dietz_recovers_true_return_exactly(clean):
    res, man = clean
    st = res.statements.set_index("statement_id")
    st = st[st.beginning_source == "printed"]   # Schwab + Fidelity synthetic accounts
    assert len(st) > 100
    for sid, row in st.iterrows():
        assert row.dietz_return == pytest.approx(man.true_returns[sid], abs=1e-6), sid


def test_coverage_report_writes(tmp_path, dirty):
    from trackrecord.coverage import write_report, year_matrix
    res, _ = dirty
    path = write_report(res, tmp_path)
    text = path.read_text()
    assert "TDA_1003: 2007-01-01 → 2007-12-31" in text
    mat = year_matrix(res.statements)
    assert mat.at["TDA_1003", 2007] == "-"
    assert mat.at["TDA_1006", 2005] == "U"
    assert mat.at["SCHW_1001", 2003] == "X"
    assert mat.at["FIDE_1008", 2000] == ""
