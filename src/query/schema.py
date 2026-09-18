"""
Defines the strict JSON shape the LLM must produce for every natural
language question. We never let the LLM generate or execute pandas/SQL
code directly -- it only picks values that fit this schema, and our own
runner.py is the only thing that ever touches the DataFrame. This keeps
the system safe, testable, and debuggable independent of the LLM.
"""

from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator

TEXT_COLUMNS = ["ticket_id", "category", "priority", "status", "agent_id", "issue_summary"]
NUMERIC_COLUMNS = ["response_time_hrs", "resolution_time_hrs", "customer_rating"]
DATE_COLUMNS = ["created_at"]

ALLOWED_COLUMNS = TEXT_COLUMNS + NUMERIC_COLUMNS + DATE_COLUMNS
ALLOWED_OPERATORS = ["eq", "neq", "gt", "gte", "lt", "lte", "in", "not_in", "contains"]
ALLOWED_METRICS = ["count", "average", "sum", "min", "max", "list"]
FilterValue = Union[str, int, float, List[Union[str, int, float]]]


class Filter(BaseModel):
    column: str
    operator: str
    value: FilterValue

    @field_validator("column")
    @classmethod
    def check_column(cls, v: str) -> str:
        if v not in ALLOWED_COLUMNS:
            raise ValueError(f"Unknown column '{v}'. Allowed: {ALLOWED_COLUMNS}")
        return v

    @field_validator("operator")
    @classmethod
    def check_operator(cls, v: str) -> str:
        if v not in ALLOWED_OPERATORS:
            raise ValueError(f"Unknown operator '{v}'. Allowed: {ALLOWED_OPERATORS}")
        return v


class QuerySpec(BaseModel):
    """
    Examples:
      "How many tickets are currently open?"
        -> metric=count, filters=[{column:status, operator:eq, value:Open}]

      "Which agent has the lowest average customer rating?"
        -> metric=average, metric_column=customer_rating,
           group_by=agent_id, sort=asc, limit=1

      "Show me all Critical tickets not resolved within 12 hours."
        -> metric=list, filters=[{priority eq Critical}, {resolution_time_hrs gt 12}]
    """

    metric: Literal["count", "average", "sum", "min", "max", "list"]
    metric_column: Optional[str] = None
    filters: List[Filter] = Field(default_factory=list)
    group_by: Optional[str] = None
    sort: Optional[Literal["asc", "desc"]] = None
    limit: Optional[int] = None

    @field_validator("metric_column", "group_by")
    @classmethod
    def check_column_field(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in ALLOWED_COLUMNS:
            raise ValueError(f"Unknown column '{v}'. Allowed: {ALLOWED_COLUMNS}")
        return v

    @field_validator("limit")
    @classmethod
    def check_limit(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v <= 0:
            raise ValueError("limit must be a positive integer")
        return v
