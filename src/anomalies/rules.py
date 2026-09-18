"""
Rule-based anomaly detection over the ticket data.

Deliberately NOT machine-learning-based: with ~500 rows and a fixed
schema, simple, explainable thresholds are more defensible and easier
to reason about than a trained model. No LLM call happens here, so
this module runs instantly and is fully unit-testable offline.

Design note on "now": the CSV is a static historical snapshot, so
"stale unresolved tickets" is measured relative to the most recent
ticket's created_at in the dataset (not the real wall-clock date),
so results stay meaningful no matter when this is run. See README
for how to point this at real wall-clock time in production.
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
