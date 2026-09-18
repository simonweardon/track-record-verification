"""Uploaded CSV shapes: what a broker's own export looks like, and what can be made of it."""
import pytest

from trackrecord import uploads as UP

ROBINHOOD_ACTIVITY = (
    '"Activity Date","Process Date","Settle Date","Instrument","Description","Trans Code","Quantity","Price","Amount"\n'
    '"7/31/2024","7/31/2024","7/31/2024","","Interest Payment","INT","","","$1.10"\n'
    '"7/22/2024","7/22/2024","7/22/2024","","ACH Withdrawal","ACH","","","($200.00)"\n'
    '"7/15/2024","7/15/2024","7/17/2024","AAPL","Apple Inc.","Buy","3","$210.00","($630.00)"\n'
    '"7/3/2024","7/3/2024","7/3/2024","","ACH Deposit","ACH","","","$500.00"\n'
).encode()


def _months(n=14):
    return [f"2023-{m:02d}-28" if m <= 12 else f"2024-{m - 12:02d}-28" for m in range(1, n + 1)]


def test_transaction_export_is_explained_not_just_rejected(tmp_path):
    for fn in (UP.from_returns, UP.from_values):
        with pytest.raises(UP.UploadError) as e:
            fn("activity.csv", ROBINHOOD_ACTIVITY, tmp_path / "never", "x")
        msg = str(e.value)
        assert "lists individual transactions" in msg and "Robinhood" in msg
        assert "Statement PDFs" in msg


def test_column_names_brokers_actually_use(tmp_path):
    rows = "\n".join(f"{d},{1000 + i * 10}.00,0" for i, d in enumerate(_months()))
    data = ("Activity Date,Portfolio Value,Net Deposits\n" + rows + "\n").encode()
    info = UP.from_values("export.csv", data, tmp_path / "v", "x")
    assert info["shape"] == "values" and info["periods"] == 14 and info["grid"] == "M"


def test_a_plain_return_file_still_loads(tmp_path):
    rows = "\n".join(f"{d},1.2" for d in _months())
    info = UP.from_returns("returns.csv", ("date,return\n" + rows + "\n").encode(), tmp_path / "r", "x")
    assert info["shape"] == "returns" and info["periods"] == 14
