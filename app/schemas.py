"""Pydantic request/response models — the API contract."""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

Intent = Literal["search", "discovery", "comparison", "customer_support"]
ActionType = Literal["recommend", "clarify", "route_to_partner", "support_handoff", "compare"]


class AssistRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="Raw user query (de/en)")
    max_results: int = Field(5, ge=1, le=20)
    user_id: str = Field("anon", min_length=1, max_length=64,
                         description="Stable id for the per-user interest context")
    llm_mode: Literal["auto", "always", "off"] = Field(
        "auto", description="auto: LLM only when rules can't parse; always: LLM on every query; off: rules only")
    model: Optional[Literal["gemini-2.5-flash-lite", "gemini-2.5-flash",
                            "gemini-3.1-flash-lite", "gemini-3.5-flash"]] = Field(
        None, description="Gemini model tier override (default: gemini-2.5-flash-lite)")


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
    engine: str = "classifier"  # "classifier" or "classifier+<llm-model>" when the LLM assisted
    user_context: Optional[dict] = None  # {user_id, interests: {category: percent}}
    latency_ms: float
