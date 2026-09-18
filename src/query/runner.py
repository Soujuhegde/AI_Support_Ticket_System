"""
Executes a validated QuerySpec against the ticket DataFrame.

This is the ONLY place that actually touches the data for NL queries.
The LLM never runs code against the DataFrame directly -- it only
produces a QuerySpec, which this module interprets safely.
"""

from typing import Any

import pandas as pd

from src.query.schema import QuerySpec

_LIST_PREVIEW_COLUMNS = [
    "ticket_id", "created_at", "category", "priority", "status",
    "agent_id", "customer_rating", "issue_summary",
]


def _apply_filters(df: pd.DataFrame, spec: QuerySpec) -> pd.DataFrame:
    for f in spec.filters:
        col, op, val = f.column, f.operator, f.value

        if col == "created_at":
            val = pd.to_datetime(val) if not isinstance(val, list) else [pd.to_datetime(v) for v in val]

        series = df[col]

        if op == "eq":
            df = df[series == val]
        elif op == "neq":
            df = df[series != val]
        elif op == "gt":
            df = df[series > val]
        elif op == "gte":
            df = df[series >= val]
        elif op == "lt":
            df = df[series < val]
        elif op == "lte":
            df = df[series <= val]
        elif op == "in":
            values = val if isinstance(val, list) else [val]
            df = df[series.isin(values)]
        elif op == "not_in":
            values = val if isinstance(val, list) else [val]
            df = df[~series.isin(values)]
        elif op == "contains":
            df = df[series.astype(str).str.contains(str(val), case=False, na=False)]

    return df


def _compute_metric(series: pd.Series, metric: str) -> Any:
    if metric == "count":
        return int(series.shape[0]) if hasattr(series, "shape") else int(len(series))
    if metric == "average":
        return round(float(series.mean()), 2) if len(series) else None
    if metric == "sum":
        return round(float(series.sum()), 2) if len(series) else None
    if metric == "min":
        return round(float(series.min()), 2) if len(series) else None
    if metric == "max":
        return round(float(series.max()), 2) if len(series) else None
    raise ValueError(f"Unsupported metric for aggregation: {metric}")


def run_query(df: pd.DataFrame, spec: QuerySpec) -> dict:
    """
    Apply a validated QuerySpec to the DataFrame and return a result dict:
      {"result": <value or list>, "answer": "<short natural language answer>"}
    """
    filtered = _apply_filters(df, spec)

    # --- Grouped query (e.g. "which agent has the lowest average rating") ---
    if spec.group_by:
        grouped = filtered.groupby(spec.group_by)

        if spec.metric == "count":
            result_series = grouped.size()
        else:
            if not spec.metric_column:
                raise ValueError("metric_column is required for average/sum/min/max")
            agg_map = {"average": "mean", "sum": "sum", "min": "min", "max": "max"}
            result_series = grouped[spec.metric_column].agg(agg_map[spec.metric])
            result_series = result_series.round(2)

        if spec.sort:
            result_series = result_series.sort_values(ascending=(spec.sort == "asc"))
        if spec.limit:
            result_series = result_series.head(spec.limit)

        result = [
            {spec.group_by: idx, "value": (None if pd.isna(val) else float(val))}
            for idx, val in result_series.items()
        ]
        answer = _phrase_grouped_answer(spec, result)
        return {"result": result, "answer": answer}

    # --- Row listing (e.g. "show me all Critical tickets...") ---
    if spec.metric == "list":
        limited = filtered.head(spec.limit) if spec.limit else filtered
        columns = [c for c in _LIST_PREVIEW_COLUMNS if c in limited.columns]
        records = limited[columns].copy()
        records["created_at"] = records["created_at"].astype(str)
        rows = records.to_dict("records")
        answer = f"Found {len(filtered)} matching ticket(s)."
        if spec.limit and len(filtered) > spec.limit:
            answer += f" Showing the first {spec.limit}."
        return {"result": rows, "answer": answer}

    # --- Single aggregate (e.g. "how many tickets are open?") ---
    if spec.metric == "count":
        value = _compute_metric(filtered, "count")
    else:
        if not spec.metric_column:
            raise ValueError("metric_column is required for average/sum/min/max")
        value = _compute_metric(filtered[spec.metric_column].dropna(), spec.metric)

    answer = _phrase_single_answer(spec, value)
    return {"result": value, "answer": answer}


def _phrase_single_answer(spec: QuerySpec, value: Any) -> str:
    if spec.metric == "count":
        return f"{value} ticket(s) match your query."
    metric_word = {"average": "average", "sum": "total", "min": "minimum", "max": "maximum"}[spec.metric]
    return f"The {metric_word} {spec.metric_column} is {value}."


def _phrase_grouped_answer(spec: QuerySpec, rows: list[dict]) -> str:
    if not rows:
        return "No matching groups found."
    top = rows[0]
    group_value = top[spec.group_by]
    metric_word = {"count": "count", "average": "average", "sum": "total", "min": "minimum", "max": "maximum"}[spec.metric]
    subject = f"{spec.metric_column} " if spec.metric_column else ""
    return f"{group_value} has the {'lowest' if spec.sort == 'asc' else 'highest'} {metric_word} {subject}({top['value']})."
