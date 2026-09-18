"""
Central configuration for TicketMind.

Loads settings from a .env file (or real environment variables) and
validates that everything the app needs is actually present. The app
is designed to FAIL FAST: if GROQ_API_KEY is missing, nothing starts
(no API, no UI) rather than starting and breaking later mid-request.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load variables from a .env file if one exists (local dev).
# In Docker, variables are usually injected directly via env_file/environment.
load_dotenv()

# --- LLM settings -----------------------------------------------------
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").strip()

# --- Data settings ------------------------------------------------------
CSV_PATH: str = os.getenv("CSV_PATH", "data/support_tickets.csv").strip()

# --- Anomaly detection thresholds --------------------------------------
ANOMALY_RESOLUTION_PERCENTILE: float = float(
    os.getenv("ANOMALY_RESOLUTION_PERCENTILE", "90")
)
ANOMALY_STALE_HOURS: float = float(os.getenv("ANOMALY_STALE_HOURS", "24"))


def validate() -> None:
    """
    Raise a clear, actionable error if required configuration is missing.
    Called once at startup by both the API and the UI.
    """
    problems = []

    if not GROQ_API_KEY:
        problems.append(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your "
            "free Groq API key from https://console.groq.com/keys"
        )

    if not Path(CSV_PATH).exists():
        problems.append(f"CSV file not found at '{CSV_PATH}'.")

    if problems:
        raise RuntimeError(
            "TicketMind cannot start due to missing configuration:\n- "
            + "\n- ".join(problems)
        )
