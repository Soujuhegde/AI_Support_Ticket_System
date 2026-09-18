"""Tests for src/data/loader.py -- no LLM or network needed."""

import pandas as pd
import pytest

from src.data.loader import load_tickets, EXPECTED_COLUMNS


@pytest.fixture
def df():
    return load_tickets("data/support_tickets.csv")


def test_loads_expected_row_count(df):
    assert len(df) == 500


def test_has_all_expected_columns(df):
    assert set(EXPECTED_COLUMNS).issubset(set(df.columns))


def test_created_at_is_datetime(df):
    assert pd.api.types.is_datetime64_any_dtype(df["created_at"])


def test_numeric_columns_are_numeric(df):
    assert pd.api.types.is_float_dtype(df["response_time_hrs"])
    assert pd.api.types.is_float_dtype(df["resolution_time_hrs"])


def test_unresolved_tickets_have_null_resolution_time(df):
    unresolved = df[df["status"].isin(["Open", "Escalated"])]
    # At least some unresolved tickets should have a null resolution time
    assert unresolved["resolution_time_hrs"].isna().any()


def test_missing_column_raises(tmp_path):
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("ticket_id,category\nTKT-001,Billing\n")
    with pytest.raises(ValueError):
        load_tickets(str(bad_csv))
