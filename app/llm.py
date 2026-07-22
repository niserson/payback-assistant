"""Optional Gemini-backed query understanding for what rules can't parse.

Two serving backends, auto-selected:
  - Vertex AI (preferred, used on Cloud Run): set VERTEX_PROJECT. Auth is the
    runtime service account via the GCE/Cloud Run metadata server - no secrets.
  - AI Studio (local dev): set GEMINI_API_KEY.

Inference-latency engineering (measured, see /performance-report):
  - thinkingBudget=0 for the 2.5 family: they reason by default; disabling hidden
    thinking cut gemini-2.5-flash from ~2.9s to ~0.9s per call.
  - Compact wire schema (single-letter keys) + maxOutputTokens cap: decode time
    dominates, so fewer output tokens is the cheapest speedup.
  - Compact prompt (~60% fewer input tokens than v1) with few-shot examples.
  - Regional Vertex endpoint (europe-west4, close to the Cloud Run region) for the
    2.5 family: ~250ms less than the global endpoint; 3.x models are global-only.
  - Small TTL response cache: identical (query, model, context) repeats are free.
  - Hard timeout + one retry, falling back to the classifier path: the API never blocks on a
    slow model.
"""

import json
import logging
import os
import time
from collections import OrderedDict
from typing import Optional

import httpx

log = logging.getLogger("assistant.llm")

_AISTUDIO_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
_VERTEX_GLOBAL_URL = ("https://aiplatform.googleapis.com/v1/projects/{project}/locations/global/"
                      "publishers/google/models/{model}:generateContent")
_VERTEX_REGIONAL_URL = ("https://{loc}-aiplatform.googleapis.com/v1/projects/{project}/locations/"
                        "{loc}/publishers/google/models/{model}:generateContent")
_METADATA_TOKEN_URL = ("http://metadata.google.internal/computeMetadata/v1/instance/"
                       "service-accounts/default/token")
_TIMEOUT_S = float(os.getenv("LLM_TIMEOUT_S", "6"))
_INTENTS = {"search", "discovery", "comparison", "customer_support"}
_PARTNERS = {"dm", "edeka", "amazon"}

# Validated against Vertex AI (2.0-flash* are not served there; 3.x are global-only).
ALLOWED_MODELS = (
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash",
)

_token_cache = {"value": None, "expires": 0.0}

# TTL response cache: repeated (model, query, context) calls skip inference entirely.
_CACHE_TTL_S = 300
_CACHE_MAX = 512
_response_cache: "OrderedDict[tuple, tuple]" = OrderedDict()

# Compact wire schema: single-letter keys keep decode short; mapped back to full
# field names in _validate so callers never see the wire format.
_PROMPT = """You classify retail queries for a German loyalty app. Partners: dm (drugstore), edeka (grocery), amazon (general merchandise).
Reply ONLY with JSON (use JSON null, never the string "null"): {{"i":"search|discovery|comparison|customer_support","l":"de|en","p":"dm|edeka|amazon"|null,"t":"German search keywords"|null,"c":"clarifying question"|null}}
Rules:
- l = language of the query. p only if the user names a shop.
- If concrete products are implied, t = short GERMAN base-product keywords (always translate; map dishes/use-cases to supermarket products; drop audience words). Ex: "spiegelei fürs frühstück"->"eier butter frühstück"; "pancakes for the kids"->"mehl eier milch zucker".
- Shopping lists (items joined by commas/und/and) are always specific.
- Only if truly vague: t = null, c = ONE short question in the query's language.
{context_block}Query: "{query}"
"""

_CONTEXT_BLOCK = """User interests (weight ~30% vs query 70%): {profile}. Resolve vague queries toward dominant categories, e.g. "Baby & Kind 80%"+"creme"->"wundschutz creme baby"; "Sport & Fitness 70%"+"was für draußen"->"campingstuhl rucksack trinkflasche". Never contradict explicit intent.
"""


def backend() -> Optional[str]:
    if os.getenv("VERTEX_PROJECT"):
        return "vertex-ai"
    if os.getenv("GEMINI_API_KEY"):
        return "ai-studio"
    return None


def available() -> bool:
    return backend() is not None


def model_name(override: Optional[str] = None) -> str:
    if override in ALLOWED_MODELS:
        return override
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


def _vertex_url(model: str) -> str:
    project = os.environ["VERTEX_PROJECT"]
    location = os.getenv("VERTEX_LOCATION", "europe-west4")
    # 2.5 family is served regionally (lower RTT from Cloud Run); 3.x is global-only.
    if model.startswith("gemini-2.5") and location != "global":
        return _VERTEX_REGIONAL_URL.format(project=project, model=model, loc=location)
    return _VERTEX_GLOBAL_URL.format(project=project, model=model)


def _request(body: dict, model: Optional[str] = None) -> httpx.Response:
    resolved = model_name(model)
    if backend() == "vertex-ai":
        url = _vertex_url(resolved)
        headers = {"Authorization": f"Bearer {_metadata_token()}"}
    else:
        url = _AISTUDIO_URL.format(model=resolved)
        headers = {"x-goog-api-key": os.environ["GEMINI_API_KEY"]}
    return httpx.post(url, headers=headers, json=body, timeout=_TIMEOUT_S)


def _generation_config(model: str) -> dict:
    config = {"responseMimeType": "application/json", "temperature": 0, "maxOutputTokens": 96}
    if model.startswith("gemini-2.5"):
        # 2.5 models spend hidden reasoning tokens by default — measured ~2s extra
        # on gemini-2.5-flash. Classification needs none of it.
        config["thinkingConfig"] = {"thinkingBudget": 0}
    return config


def classify(query: str, user_context: Optional[str] = None,
             model: Optional[str] = None) -> Optional[dict]:
    """Returns a validated understanding dict, or None (caller falls back to the classifier)."""
    if not available():
        return None
    resolved = model_name(model)
    cache_key = (resolved, query.strip().lower(), user_context or "")
    now = time.time()
    cached = _response_cache.get(cache_key)
    if cached and now - cached[0] < _CACHE_TTL_S:
        _response_cache.move_to_end(cache_key)
        return dict(cached[1]) if cached[1] else None

    context_block = _CONTEXT_BLOCK.format(profile=user_context) if user_context else ""
    prompt = _PROMPT.format(query=query.replace('"', "'"), context_block=context_block)
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": _generation_config(resolved),
    }
    for attempt in (1, 2):  # one retry: transient 5xx happen
        try:
            response = _request(body, model)
            if response.status_code >= 500 and attempt == 1:
                time.sleep(0.3)
                continue
            response.raise_for_status()
            text = response.json()["candidates"][0]["content"]["parts"][0]["text"]
            result = _validate(json.loads(text))
            _response_cache[cache_key] = (now, dict(result) if result else None)
            while len(_response_cache) > _CACHE_MAX:
                _response_cache.popitem(last=False)
            return result
        except Exception as exc:  # noqa: BLE001 — any LLM failure must not break the API
            if attempt == 2:
                log.warning("gemini unavailable, falling back to classifier: %s", exc)
                return None
            time.sleep(0.3)
    return None


def _validate(data: dict) -> Optional[dict]:
    if not isinstance(data, dict):
        return None
    # Map compact wire keys to the stable field names callers use.
    mapped = {
        "intent": data.get("i", data.get("intent")),
        "language": data.get("l", data.get("language")),
        "partner": data.get("p", data.get("partner")),
        "search_terms": data.get("t", data.get("search_terms")),
        "clarifying_question": data.get("c", data.get("clarifying_question")),
    }
    if mapped["intent"] not in _INTENTS:
        return None
    if mapped["language"] not in ("de", "en"):
        mapped["language"] = "de"
    if mapped["partner"] not in _PARTNERS:
        mapped["partner"] = None
    for field in ("search_terms", "clarifying_question"):
        value = mapped[field]
        cleaned = value.strip() if isinstance(value, str) else None
        # models occasionally emit the STRING "null"/"none" instead of JSON null
        mapped[field] = None if not cleaned or cleaned.lower() in ("null", "none") else cleaned
    if not mapped["search_terms"] and not mapped["clarifying_question"] and mapped["intent"] not in (
        "customer_support",
    ):
        return None  # nothing actionable came back
    return mapped
