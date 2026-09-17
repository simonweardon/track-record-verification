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


def _robinhood(year, month, last, opening, closing, deposit=0.0, withdrawal=0.0):
    """A month of Robinhood-shaped statement text: opening/closing columns, dated activity."""
    lines = ["Robinhood Securities, LLC", "500 Colonial Center Parkway, Suite 100", "Lake Mary, FL 32746",
             "ACCOUNT STATEMENT", f"{month:02d}/01/{year} to {month:02d}/{last}/{year}",
             "JANE DOE", "123 MAIN ST", "Account Number: 123456789", "Account Type: Individual",
             "PORTFOLIO SUMMARY", "Opening Balance Closing Balance",
             f"Securities ${opening - 500:,.2f} ${closing - 500:,.2f}", "Options $0.00 $0.00",
             "Cash and Cash Equivalents $500.00 $500.00", f"Total ${opening:,.2f} ${closing:,.2f}",
             "INCOME AND EXPENSE SUMMARY", "This Period Year to Date", "Dividends $12.00 $85.00",
             "Interest $1.10 $9.40", "ACCOUNT ACTIVITY",
             "Settle Date Description Symbol Acct Type Transaction Qty Price Amount"]
    if deposit:
        lines.append(f"{month:02d}/05/{year} {month:02d}/05/{year} ACH Deposit ACH Individual Deposit ${deposit:,.2f}")
    if withdrawal:
        lines.append(f"{month:02d}/20/{year} {month:02d}/20/{year} ACH Withdrawal ACH Individual Withdrawal (${withdrawal:,.2f})")
    lines += [f"{month:02d}/15/{year} Buy AAPL Individual Buy 3 $210.00 ($630.00)",
              f"{month:02d}/28/{year} AAPL Cash Dividend Dividend $12.00"]
    return lines


def test_reads_opening_and_closing_columns():
    p = PS.parse_statement_text(PS.extract_text(_pdf(_robinhood(2024, 7, 31, 100000.0, 101505.0, 1000.0, 500.0))))
    assert str(p["period_start"]) == "2024-07-01" and str(p["period_end"]) == "2024-07-31"
    assert p["custodian"] == "Robinhood" and p["account_last4"] == "6789"
    assert p["beginning_value"] == 100000.0 and p["ending_value"] == 101505.0
    assert not p["missing"]


def test_activity_list_gives_dated_transfers_and_skips_trades():
    text = PS.extract_text(_pdf(_robinhood(2024, 7, 31, 100000.0, 101505.0, 1000.0, 500.0)))
    xf = PS.parse_cash_transfers(text)
    assert [t["amount"] for t in xf] == [1000.0, -500.0]
    assert [str(t["date"]) for t in xf] == ["2024-07-05", "2024-07-20"]
    import datetime as _dt
    assert PS.parse_cash_transfers(text, _dt.date(2024, 7, 10), _dt.date(2024, 7, 31)) == xf[1:]


def test_robinhood_statements_build_a_dataset(tmp_path):
    import calendar
    import pandas as pd
    files, value = [], 100000.0
    for i in range(13):
        year, month = 2023 + (i + 6) // 12, (i + 6) % 12 + 1
        last = calendar.monthrange(year, month)[1]
        dep = 1000.0 if i % 3 == 0 else 0.0
        closing = round((value + dep) * 1.01, 2)
        files.append((f"{year}-{month:02d}.pdf", _pdf(_robinhood(year, month, last, value, closing, dep))))
        value = closing
    info, reports = PS.from_pdfs(files, tmp_path, "Robinhood main")
    assert info["parsed"] == 13 and info["skipped"] == 0 and info["grid"] == "M"
    st = pd.read_csv(tmp_path / "statements.csv")
    assert len(st) == 13 and st.custodian.eq("Robinhood").all()
    # every month after the first chains onto the one before it, which is what lets the
    # pipeline verify the record rather than take it on trust
    assert (st.beginning_value.iloc[1:].values == st.ending_value.iloc[:-1].values).all()
    fl = pd.read_csv(tmp_path / "flows.csv")
    assert len(fl) == 5 and fl.amount.eq(1000.0).all() and fl.date.str.endswith("-05").all()


def test_total_row_beats_the_label_on_a_two_column_statement():
    # "Total Account Value $100,000.00 $105,000.00": reading the label alone would take the
    # opening figure, $100,000, as the ending value
    p = PS.parse_statement_text("Charles Schwab & Co., Inc.\nAccount Number: 1234-5678\n"
                                "Statement Period March 1, 2024 to March 31, 2024\nAccount Summary\n"
                                "Beginning Value Ending Value\nCash and Cash Investments $5,000.00 $6,000.00\n"
                                "Market Value of Investments $95,000.00 $99,000.00\n"
                                "Total Account Value $100,000.00 $105,000.00\n")
    assert p["beginning_value"] == 100000.0 and p["ending_value"] == 105000.0


def test_columns_are_read_in_the_order_the_headings_give_them():
    p = PS.parse_statement_text("Some Broker\nStatement Period: March 1, 2024 - March 31, 2024\n"
                                "Account Number: 9999-1234\nPortfolio Summary\n"
                                "Closing Balance Opening Balance\nTotal $105,000.00 $100,000.00\n")
    assert p["beginning_value"] == 100000.0 and p["ending_value"] == 105000.0


def test_single_column_statements_are_untouched():
    p = PS.parse_statement_text("Small Bank\nStatement Period: March 1, 2024 - March 31, 2024\n"
                                "Account Number: 4321-9876\nBeginning Value $10,000.00\nAdditions $0.00\n"
                                "Ending Value $10,500.00\n")
    assert p["beginning_value"] == 10000.0 and p["ending_value"] == 10500.0 and p["stated_deposits"] == 0.0
