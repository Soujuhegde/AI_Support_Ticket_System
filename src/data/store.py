"""
A single in-memory DataFrame shared by the API and the UI, so the CSV
is loaded once and both surfaces query the exact same data.
"""

import pandas as pd

from src.data.loader import load_tickets

_df: pd.DataFrame | None = None


def init_store(csv_path: str) -> None:
    """Load the CSV once. Safe to call more than once (re-loads)."""
    global _df
    _df = load_tickets(csv_path)


def get_dataframe() -> pd.DataFrame:
    """Return a copy of the loaded ticket data. Raises if not initialized."""
    if _df is None:
        raise RuntimeError(
            "Ticket data store has not been initialized. Call init_store() first."
        )
    return _df.copy()
