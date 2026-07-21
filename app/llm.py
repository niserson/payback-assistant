"""Optional Gemini-backed query understanding for what rules can't parse.

Two serving backends, auto-selected:
  - Vertex AI (preferred, used on Cloud Run): set VERTEX_PROJECT. Auth is the
    runtime service account via the GCE/Cloud Run metadata server - no secrets.
  - AI Studio (local dev): set GEMINI_API_KEY.

Cost/latency discipline: the deterministic rule path answers everything it can in
~0.2 ms; Gemini is consulted ONLY when rules find no retrievable product term in
the query (paraphrases, vague needs). Every failure mode (no key, timeout, 5xx,
malformed JSON) degrades gracefully back to the rule path — the API never breaks
because the LLM is down.
"""

import json
import logging
import os
import time
from typing import Optional

import httpx

log = logging.getLogger("assistant.llm")

_AISTUDIO_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
_VERTEX_URL = ("https://aiplatform.googleapis.com/v1/projects/{project}/locations/global/"
               "publishers/google/models/{model}:generateContent")
_METADATA_TOKEN_URL = ("http://metadata.google.internal/computeMetadata/v1/instance/"
                       "service-accounts/default/token")
_TIMEOUT_S = 6.0
_INTENTS = {"search", "discovery", "comparison", "customer_support"}
_PARTNERS = {"dm", "edeka", "amazon"}

_token_cache = {"value": None, "expires": 0.0}

_PROMPT = """You are the query-understanding module of a German retail loyalty app.
Partners: dm (drugstore), EDEKA (grocery/fresh produce), Amazon (general merchandise).
Classify the user query and reply with ONLY this JSON object:
{{
  "intent": "search" | "discovery" | "comparison" | "customer_support",
  "language": "de" | "en",
  "partner": "dm" | "edeka" | "amazon" | null,
  "search_terms": string | null,
  "clarifying_question": string | null
}}
Rules:
- "language" is the language of the query.
- "partner" only if the user names a shop.
- If the query names or clearly implies concrete products, set "search_terms" to
  short GERMAN keywords for BASE PRODUCTS a supermarket/drugstore sells (always
  translate to German; map dishes, recipes and use-cases to their ingredients or
  the products that fulfil them) and leave "clarifying_question" null.
  Example: "something to soothe my toddler's diaper rash" -> "wundschutz creme baby"
  Example: "spiegelei fürs frühstück" -> "eier butter frühstück"
  Example: "pancakes for the kids" -> "mehl eier milch zucker" (ingredients, not the
  dish; drop audience words like kids/Kinder unless the product itself is for them)
  Example: "was fürs Grillfest am Samstag" -> "bratwurst grillen"
- If the need is genuinely too vague to search, set "search_terms" null and ask ONE
  short clarifying question in the query's language.
Query: "{query}"
"""


def backend() -> Optional[str]:
    if os.getenv("VERTEX_PROJECT"):
        return "vertex-ai"
    if os.getenv("GEMINI_API_KEY"):
        return "ai-studio"
    return None


def available() -> bool:
    return backend() is not None


def model_name() -> str:
    return os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")


def _metadata_token() -> str:
    """Access token of the Cloud Run / GCE service account (cached until expiry)."""
    now = time.time()
    if _token_cache["value"] and now < _token_cache["expires"]:
        return _token_cache["value"]
    response = httpx.get(_METADATA_TOKEN_URL, headers={"Metadata-Flavor": "Google"}, timeout=3.0)
    response.raise_for_status()
    data = response.json()
    _token_cache["value"] = data["access_token"]
    _token_cache["expires"] = now + int(data.get("expires_in", 300)) - 60
    return _token_cache["value"]


def _request(body: dict) -> httpx.Response:
    if backend() == "vertex-ai":
        url = _VERTEX_URL.format(project=os.environ["VERTEX_PROJECT"], model=model_name())
        headers = {"Authorization": f"Bearer {_metadata_token()}"}
    else:
        url = _AISTUDIO_URL.format(model=model_name())
        headers = {"x-goog-api-key": os.environ["GEMINI_API_KEY"]}
    return httpx.post(url, headers=headers, json=body, timeout=_TIMEOUT_S)


def classify(query: str) -> Optional[dict]:
    """Returns a validated understanding dict, or None (caller falls back to rules)."""
    if not available():
        return None
    body = {
        "contents": [{"role": "user", "parts": [{"text": _PROMPT.format(query=query.replace('"', "'"))}]}],
        "generationConfig": {"responseMimeType": "application/json", "temperature": 0},
    }
    for attempt in (1, 2):  # one retry: transient 5xx happen
        try:
            response = _request(body)
            if response.status_code >= 500 and attempt == 1:
                time.sleep(0.3)
                continue
            response.raise_for_status()
            text = response.json()["candidates"][0]["content"]["parts"][0]["text"]
            return _validate(json.loads(text))
        except Exception as exc:  # noqa: BLE001 — any LLM failure must not break the API
            if attempt == 2:
                log.warning("gemini unavailable, falling back to rules: %s", exc)
                return None
            time.sleep(0.3)
    return None


def _validate(data: dict) -> Optional[dict]:
    if not isinstance(data, dict) or data.get("intent") not in _INTENTS:
        return None
    if data.get("language") not in ("de", "en"):
        data["language"] = "de"
    if data.get("partner") not in _PARTNERS:
        data["partner"] = None
    for field in ("search_terms", "clarifying_question"):
        value = data.get(field)
        data[field] = value.strip() if isinstance(value, str) and value.strip() else None
    if not data["search_terms"] and not data["clarifying_question"] and data["intent"] not in (
        "customer_support",
    ):
        return None  # nothing actionable came back
    return data
