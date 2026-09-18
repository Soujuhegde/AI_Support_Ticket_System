"""
FastAPI entrypoint for TicketMind.

Fails fast on startup if configuration (GROQ_API_KEY, CSV path) is
missing -- the API will not come up at all rather than serving broken
requests. Run with: uvicorn src.api.main:app --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI

from src import config
from src.api.routes import router
from src.data.store import init_store

app = FastAPI(
    title="TicketMind",
    description="Natural language querying and anomaly detection over customer support tickets.",
    version="1.0.0",
)

app.include_router(router)


@app.on_event("startup")
def startup() -> None:
    config.validate()  # raises RuntimeError -> app will not start without GROQ_API_KEY
    init_store(config.CSV_PATH)
