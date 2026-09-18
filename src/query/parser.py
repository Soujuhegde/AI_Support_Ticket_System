"""
Validates the LLM's raw text response against QuerySpec. LLMs
occasionally wrap JSON in markdown fences or add stray text -- this
module cleans that up before validation, and raises a clear error if
the result still doesn't fit the schema.
"""

import json
import re

from pydantic import ValidationError

from src.query.schema import QuerySpec


class QueryParseError(Exception):
    """Raised when the LLM's output cannot be turned into a valid QuerySpec."""


def _strip_code_fences(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` wrappers some models add."""
    text = text.strip()
    match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    return match.group(1) if match else text


def parse_query_spec(raw_llm_output: str) -> QuerySpec:
    """
    Parse and validate raw LLM text into a QuerySpec.
    Raises QueryParseError with a human-readable message on failure.
    """
    cleaned = _strip_code_fences(raw_llm_output)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise QueryParseError(f"LLM did not return valid JSON: {exc}") from exc

    try:
        return QuerySpec.model_validate(data)
    except ValidationError as exc:
        raise QueryParseError(f"LLM JSON did not match the expected schema: {exc}") from exc
