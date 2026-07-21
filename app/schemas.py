"""Pydantic request/response models — the API contract."""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

Intent = Literal["search", "discovery", "comparison", "customer_support"]
ActionType = Literal["recommend", "clarify", "route_to_partner", "support_handoff", "compare"]


class AssistRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="Raw user query (de/en)")
    max_results: int = Field(5, ge=1, le=20)


class Product(BaseModel):
    id: str
    partner: str
    name: str
    brand: str
    category: str
    price: float
    unit: str
    tags: List[str]
    score: float


class Action(BaseModel):
    type: ActionType
    detail: str


class AssistResponse(BaseModel):
    query: str
    language: Literal["de", "en"]
    intent: Intent
    confidence: float
    action: Action
    partner_filter: Optional[str] = None
    products: List[Product] = []
    clarifying_question: Optional[str] = None
    engine: str = "rules"  # "rules" or "rules+<llm-model>" when the LLM assisted
    latency_ms: float
