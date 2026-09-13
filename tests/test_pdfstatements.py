import pymupdf
import pytest

from trackrecord import pdfstatements as PS


def _pdf(lines):
    doc = pymupdf.open(); page = doc.new_page(); y = 72
    for l in lines:
        page.insert_text((72, y), l, fontsize=11); y += 18
    return doc.tobytes()


def test_parses_fidelity_style_summary():
    text = PS.extract_text(_pdf(["Fidelity Investments", "Statement Period: March 1, 2024 - March 31, 2024", "Account Number: X12-345678",
                                 "Account Summary", "Beginning Account Value $1,054,250.00", "Additions $0.00", "Subtractions ($20,000.00)",
                                 "Change in Investment Value -$8,250.00", "Ending Account Value $1,026,000.00"]))
    p = PS.parse_statement_text(text)
    assert str(p["period_start"]) == "2024-03-01" and str(p["period_end"]) == "2024-03-31"
    assert p["account_last4"] == "5678" and p["custodian"] == "Fidelity"
    assert p["beginning_value"] == 1054250.0 and p["ending_value"] == 1026000.0
    assert p["stated_withdrawals"] == 20000.0 and p["stated_deposits"] == 0.0 and p["stated_pnl"] == -8250.0
    assert not p["missing"]


def test_reports_missing_and_refuses_scan():
    p = PS.parse_statement_text("Some Bank\nStatement for the period ending 06/30/2024\nTotal Account Value $12,345.67\n")
    assert str(p["period_end"]) == "2024-06-30" and p["ending_value"] == 12345.67
    assert "beginning_value" in p["missing"] and "account number" in p["missing"]
    scan = _pdf(["x"])
    _, reps = (None, None)
    with pytest.raises(Exception):
        PS.from_pdfs([("scan.pdf", scan)], __import__("pathlib").Path("/tmp/never"), "x")


def test_negative_number_forms():
    assert PS._num("($1,234.50)") == -1234.5 and PS._num("-$1,234.50") == -1234.5 and PS._num("$-1,234.50") == -1234.5
    assert PS._num("$1,234.50") == 1234.5 and PS._num("-") == 0.0
