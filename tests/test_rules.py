"""Tests for src/anomalies/rules.py -- pure pandas logic, no LLM needed."""

import pytest

from src.data.loader import load_tickets
from src.anomalies.rules import detect_anomalies


@pytest.fixture
def df():
    return load_tickets("data/support_tickets.csv")


def test_returns_expected_keys(df):
    report = detect_anomalies(df)
    assert set(report.keys()) == {
        "reference_time",
        "resolution_time_threshold_hrs",
        "stale_hours_threshold",
        "long_resolution_time",
        "stale_high_priority",
    }


def test_long_resolution_tickets_exceed_threshold(df):
    report = detect_anomalies(df, resolution_percentile=90)
    threshold = report["resolution_time_threshold_hrs"]
    for row in report["long_resolution_time"]:
        assert row["resolution_time_hrs"] > threshold


def test_stale_tickets_are_unresolved_and_high_priority(df):
    report = detect_anomalies(df, stale_hours=24)
    for row in report["stale_high_priority"]:
        assert row["status"] in ("Open", "Escalated")
        assert row["priority"] in ("High", "Critical")
        assert row["age_hours"] > 24


def test_higher_percentile_flags_fewer_or_equal_tickets(df):
    report_90 = detect_anomalies(df, resolution_percentile=90)
    report_99 = detect_anomalies(df, resolution_percentile=99)
    assert len(report_99["long_resolution_time"]) <= len(report_90["long_resolution_time"])


def test_result_is_json_serializable(df):
    import json
    report = detect_anomalies(df)
    json.dumps(report)  # should not raise (no NaN/NaT leaking through)
