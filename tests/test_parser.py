"""Tests for src/query/parser.py -- validates LLM JSON output handling."""

import pytest

from src.query.parser import parse_query_spec, QueryParseError


def test_parses_clean_json():
    raw = '{"metric": "count", "filters": [], "group_by": null, "sort": null, "limit": null}'
    spec = parse_query_spec(raw)
    assert spec.metric == "count"


def test_strips_markdown_code_fences():
    raw = '```json\n{"metric": "count", "filters": []}\n```'
    spec = parse_query_spec(raw)
    assert spec.metric == "count"


def test_invalid_json_raises_parse_error():
    with pytest.raises(QueryParseError):
        parse_query_spec("this is not json")


def test_unknown_column_raises_parse_error():
    raw = '{"metric": "count", "filters": [{"column": "not_a_real_column", "operator": "eq", "value": "x"}]}'
    with pytest.raises(QueryParseError):
        parse_query_spec(raw)


def test_unknown_operator_raises_parse_error():
    raw = '{"metric": "count", "filters": [{"column": "status", "operator": "bogus", "value": "Open"}]}'
    with pytest.raises(QueryParseError):
        parse_query_spec(raw)


def test_invalid_metric_raises_parse_error():
    raw = '{"metric": "median", "filters": []}'
    with pytest.raises(QueryParseError):
        parse_query_spec(raw)
