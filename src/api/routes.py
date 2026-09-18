"""
The three required REST endpoints:
  GET  /health     -> liveness check
  POST /query      -> natural language question -> answer
  GET  /anomalies  -> rule-based anomaly report
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src import config
from src.anomalies.rules import detect_anomalies
from src.data.store import get_dataframe
from src.query.llm import call_llm
from src.query.parser import QueryParseError, parse_query_spec
from src.query.prompt import build_messages
from src.query.runner import run_query

router = APIRouter()


class QueryRequest(BaseModel):
    question: str


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.post("/query")
def query(request: QueryRequest) -> dict:
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="'question' must not be empty.")

    df = get_dataframe()
    reference_date = str(df["created_at"].max())

    try:
        messages = build_messages(request.question, reference_date)
        raw_output = call_llm(messages)
        spec = parse_query_spec(raw_output)
        result = run_query(df, spec)
    except QueryParseError as exc:
        raise HTTPException(status_code=422, detail=f"Could not understand the question: {exc}") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return {
        "question": request.question,
        "query_spec": spec.model_dump(),
        "answer": result["answer"],
        "result": result["result"],
    }


@router.get("/anomalies")
def anomalies() -> dict:
    df = get_dataframe()
    return detect_anomalies(
        df,
        resolution_percentile=config.ANOMALY_RESOLUTION_PERCENTILE,
        stale_hours=config.ANOMALY_STALE_HOURS,
    )
