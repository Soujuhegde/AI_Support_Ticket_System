"""Tests for src/query/runner.py -- the actual query execution logic."""

import pytest

from src.data.loader import load_tickets
from src.query.schema import QuerySpec
from src.query.runner import run_query


@pytest.fixture
def df():
    return load_tickets("data/support_tickets.csv")


def test_simple_count(df):
    spec = QuerySpec(
        metric="count",
        filters=[{"column": "status", "operator": "eq", "value": "Open"}],
    )
    result = run_query(df, spec)
    expected = (df["status"] == "Open").sum()
    assert result["result"] == expected


def test_average_with_filter(df):
    spec = QuerySpec(
        metric="average",
        metric_column="customer_rating",
        filters=[{"column": "category", "operator": "eq", "value": "Technical"}],
    )
    result = run_query(df, spec)
    expected = round(
        df[df["category"] == "Technical"]["customer_rating"].dropna().astype(float).mean(), 2
    )
    assert result["result"] == expected


def test_group_by_with_sort_and_limit(df):
    spec = QuerySpec(
        metric="average",
        metric_column="customer_rating",
        group_by="agent_id",
        sort="asc",
        limit=1,
    )
    result = run_query(df, spec)
    assert len(result["result"]) == 1
    # The single returned group should indeed be the lowest-average agent
    all_groups = df.groupby("agent_id")["customer_rating"].mean()
    assert result["result"][0]["agent_id"] == all_groups.idxmin()


def test_list_query_respects_limit(df):
    spec = QuerySpec(
        metric="list",
        filters=[{"column": "priority", "operator": "eq", "value": "Critical"}],
        limit=3,
    )
    result = run_query(df, spec)
    assert len(result["result"]) <= 3


def test_in_operator(df):
    spec = QuerySpec(
        metric="count",
        filters=[{"column": "status", "operator": "in", "value": ["Open", "Escalated"]}],
    )
    result = run_query(df, spec)
    expected = df["status"].isin(["Open", "Escalated"]).sum()
    assert result["result"] == expected
