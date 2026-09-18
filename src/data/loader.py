"""
Loads support_tickets.csv into a clean pandas DataFrame.

Responsible only for reading + type-cleaning. No business logic here
(that lives in query/runner.py and anomalies/rules.py) so this module
stays trivial to unit test.
"""

import pandas as pd

EXPECTED_COLUMNS = [
    "ticket_id",
    "created_at",
    "category",
    "priority",
    "status",
    "response_time_hrs",
    "resolution_time_hrs",
    "agent_id",
    "customer_rating",
    "issue_summary",
]


def load_tickets(csv_path: str) -> pd.DataFrame:
    """
    Read the ticket CSV and return a DataFrame with correct dtypes.

    - created_at -> parsed datetime
    - response_time_hrs / resolution_time_hrs -> float (NaN if missing,
      which is expected for unresolved tickets)
    - customer_rating -> nullable integer (NaN if missing)
    """
    df = pd.read_csv(csv_path)

    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing expected columns: {sorted(missing)}")

    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    df["response_time_hrs"] = pd.to_numeric(
        df["response_time_hrs"], errors="coerce"
    )
    df["resolution_time_hrs"] = pd.to_numeric(
        df["resolution_time_hrs"], errors="coerce"
    )
    df["customer_rating"] = pd.to_numeric(
        df["customer_rating"], errors="coerce"
    ).astype("Int64")

    for col in ["category", "priority", "status", "agent_id"]:
        df[col] = df[col].astype(str).str.strip()

    return df
