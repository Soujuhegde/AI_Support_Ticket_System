"""
Rule-based detection to find delayed and problem tickets.

Why simple rules instead of AI / Machine Learning?
With only 500 tickets, simple mathematical limits (like finding the slowest 10%
or tickets open for >24 hours) are 100% explainable, run instantly, and are easy
to test without needing complex AI models.

Why we use the latest ticket date as "current time":
The CSV dataset is from early 2024. If we compared against today's real date,
every ticket would look years old. So we treat the newest ticket in the CSV
as "current time" to get accurate, realistic results.
"""

from typing import Optional

import pandas as pd

_PREVIEW_COLUMNS = [
    "ticket_id", "created_at", "category", "priority", "status",
    "agent_id", "resolution_time_hrs", "customer_rating",
]


def detect_anomalies(
    df: pd.DataFrame,
    resolution_percentile: float = 90,
    stale_hours: float = 24,
    reference_time: Optional[pd.Timestamp] = None,
) -> dict:
    """
    Returns a dict with two anomaly categories plus the parameters used:

      long_resolution_time: resolved tickets whose resolution_time_hrs
        exceeds the given percentile among resolved tickets.

      stale_high_priority: High/Critical tickets still Open/Escalated
        and older than stale_hours (relative to reference_time).
    """
    if reference_time is None:
        reference_time = df["created_at"].max()

    resolved = df[df["resolution_time_hrs"].notna()]
    threshold = None
    long_resolution_rows = resolved.iloc[0:0]
    if not resolved.empty:
        threshold = round(
            float(resolved["resolution_time_hrs"].quantile(resolution_percentile / 100)), 2
        )
        long_resolution_rows = resolved[resolved["resolution_time_hrs"] > threshold]

    unresolved_high = df[
        df["status"].isin(["Open", "Escalated"])
        & df["priority"].isin(["High", "Critical"])
    ].copy()
    unresolved_high["age_hours"] = (
        (reference_time - unresolved_high["created_at"]).dt.total_seconds() / 3600
    ).round(1)
    stale_rows = unresolved_high[unresolved_high["age_hours"] > stale_hours]

    return {
        "reference_time": str(reference_time),
        "resolution_time_threshold_hrs": threshold,
        "stale_hours_threshold": stale_hours,
        "long_resolution_time": _to_preview(long_resolution_rows),
        "stale_high_priority": _to_preview(stale_rows, extra_cols=["age_hours"]),
    }


def _to_preview(df: pd.DataFrame, extra_cols: Optional[list[str]] = None) -> list[dict]:
    cols = [c for c in _PREVIEW_COLUMNS if c in df.columns] + (extra_cols or [])
    out = df[cols].copy()
    out["created_at"] = out["created_at"].astype(str)
    # Replace NaN/NaT/pd.NA with None so the result is valid, standard JSON.
    out = out.astype(object).where(out.notna(), None)
    return out.to_dict("records")
