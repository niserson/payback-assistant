"""Next-Best-Action policy: turns (intent, language, retrieval) into one response.

Decision table per the challenge:
  specific      -> execute cross-catalog search (recommend)
  vague         -> ask a clarifying question (+ profile suggestions if history exists)
  navigational  -> route to a partner-scoped search
  support       -> hand off to customer service
  comparison    -> retrieve candidates and present a side-by-side

Understanding is a learned classifier (~0.3 ms); the LLM (app.llm) is consulted
when the query contains tokens the index can't ground — it translates the need
into catalog terms or asks a clarifying question, with silent fallback to the
deterministic path if the LLM is unavailable.
"""

import time
from typing import Optional

from . import context, llm, semantic
from .intent import IntentResult, detect
from .retrieval import BM25Index
from .schemas import Action, AssistResponse, Product

# Response copy (UI text, not classification logic).
_CLARIFY = {
    "de": "Kannst du das etwas eingrenzen? Suchst du z.B. Drogerie-Artikel, Lebensmittel oder etwas anderes — und bevorzugst du Bio-Marken?",
    "en": "Could you narrow that down a bit? For example: drugstore items, groceries, or something else — and do you prefer organic brands?",
}
_SUPPORT_MSG = {
    "de": "Das klingt nach einem Anliegen für unseren Kundenservice. Ich leite dich weiter: PAYBACK Service unter 089 / 996 331 60 oder im Hilfe-Center der App (Konto → Hilfe).",
    "en": "That sounds like a case for our customer service. Routing you now: PAYBACK Service at +49 89 996 331 60 or via the in-app Help Center (Account → Help).",
}


def _profile_suggestions(index: BM25Index, profile: dict, k: int) -> list:
    """Popularity-ranked picks from the user's top interest categories.

    Deterministic backstop for vague queries: the clarifying question is still
    asked, but a user with history also sees suggestions from the categories
    they demonstrably care about (a cold-start user gets the pure question).
    """
    top_categories = sorted(profile.items(), key=lambda kv: -kv[1])[:3]
    picks = []
    for category, share in top_categories:
        items = sorted((p for p in index.products if p["category"] == category),
                       key=lambda p: -p["popularity"])
        for product in items[: max(1, k // len(top_categories))]:
            item = {key: value for key, value in product.items() if key != "popularity"}
            item["score"] = round(product["popularity"] * share / 100, 3)
            picks.append(item)
    picks.sort(key=lambda p: -p["score"])
    return picks[:k]


def handle(query: str, index: BM25Index, max_results: int = 5, user_id: str = "anon",
           llm_mode: str = "auto", model: Optional[str] = None) -> AssistResponse:
    start = time.perf_counter()
    result: IntentResult = detect(query, index.vocabulary())
    lang = result.language
    engine = "classifier"
    search_query = query
    llm_clarify: Optional[str] = None
    # Interest profile BEFORE this query: 30% weight in the LLM prompt and as a
    # category rank boost in retrieval.
    profile = context.interests(user_id) or None

    # auto: escalate to the LLM when the classifier's output can't be grounded —
    # nothing retrievable, or some content tokens unknown to the index (e.g.
    # "spiegelei fur fruhstuck": 'frühstück' matches but 'spiegelei' doesn't).
    # always: every query goes through the LLM. off: deterministic only.
    needs_llm = (not result.is_specific) or bool(result.unknown_tokens)
    use_llm = llm.available() and llm_mode != "off" and (
        llm_mode == "always" or (needs_llm and result.intent in ("search", "discovery")))
    if use_llm:
        understood = llm.classify(query, user_context=context.prompt_context(user_id), model=model)
        if understood:
            engine = f"classifier+{llm.model_name(model)}@{llm.backend()}"
            lang = understood["language"]
            result.intent = understood["intent"]
            result.partner = understood["partner"] or result.partner
            if understood["search_terms"]:
                search_query = understood["search_terms"]
                result.is_specific, result.is_vague = True, False
            elif understood["clarifying_question"]:
                llm_clarify = understood["clarifying_question"]

    # Semantic net: when the LLM did not resolve the query (off, unavailable, or
    # timed out) and the lexical index can't ground it, fall back to cosine top-k
    # over EmbeddingGemma product embeddings — deterministic, ~0 extra cost since
    # the query embedding already exists. Confidence-gated (SIM_THRESHOLD) so
    # out-of-domain queries still get a clarifying question.
    semantic_products: list = []
    if (engine == "classifier" and not llm_clarify and result.embedding is not None
            and needs_llm and not result.is_vague
            and result.intent in ("search", "discovery")):
        hits = semantic.topk(result.embedding, k=max_results)
        if hits and hits[0][1] >= semantic.SIM_THRESHOLD:
            result.is_specific = True  # grounded semantically; engine marked only if used
            semantic_products = [
                {**{k: v for k, v in p.items() if k != "popularity"}, "score": round(sim, 3)}
                for p, sim in hits
            ]

    def run_search(partner=None, k=max_results):
        return index.search(search_query, top_k=k, partner=partner, interests=profile,
                            price_sensitive=result.price_sensitive)

    products: list = []
    action: Action
    clarifying: Optional[str] = None

    if result.intent == "customer_support":
        action = Action(type="support_handoff", detail=_SUPPORT_MSG[lang])

    elif result.intent == "comparison":
        products = run_search(k=max(2, max_results))
        detail = ("Vergleich der besten Treffer über alle Partner (Preis pro Einheit beachten)."
                  if lang == "de" else
                  "Side-by-side of the top matches across all partners (check price per unit).")
        action = Action(type="compare", detail=detail)

    elif result.partner is not None:
        # Navigational: partner-scoped search; with no residual terms, plain routing.
        products = run_search(partner=result.partner)
        detail = (f"Weiterleitung zur Partner-Suche: {result.partner}."
                  if lang == "de" else f"Routing to partner-specific search: {result.partner}.")
        action = Action(type="route_to_partner", detail=detail)

    elif llm_clarify:
        clarifying = llm_clarify
        action = Action(type="clarify", detail=clarifying)

    elif (result.is_vague and result.intent == "discovery") or not result.is_specific:
        clarifying = _CLARIFY[lang]
        action = Action(type="clarify", detail=clarifying)

    else:  # search or grounded discovery
        products = run_search()
        if not products and semantic_products:
            products = semantic_products
            engine = "classifier+semantic"
        if products:
            detail = ("Suchergebnisse über alle Partner-Kataloge." if lang == "de"
                      else "Search results across all partner catalogs.")
            if result.intent == "discovery":
                detail = ("Empfehlungen passend zu deiner Anfrage, partnerübergreifend."
                          if lang == "de" else
                          "Recommendations for your request across all partner catalogs.")
            action = Action(type="recommend", detail=detail)
        else:
            clarifying = _CLARIFY[lang]
            action = Action(type="clarify", detail=clarifying)

    # Vague query + existing profile: keep the clarifying question but add
    # deterministic suggestions from the user's top categories.
    suggested = False
    if action.type == "clarify" and profile and not products:
        products = _profile_suggestions(index, profile, max_results)
        suggested = bool(products)
        if suggested:
            action = Action(type="clarify", detail=action.detail + (
                " Bis dahin ein paar Ideen basierend auf deinen Interessen."
                if lang == "de" else " Meanwhile, a few ideas based on your interests."))

    # Store this query's interests, then report the updated profile back.
    # Profile-based suggestions are NOT recorded — that would be a feedback loop
    # reinforcing the profile without any user intent behind it.
    context.record(user_id, [] if suggested else [p["category"] for p in products])
    user_context = {"user_id": user_id, "interests": context.interests(user_id)}

    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    return AssistResponse(
        query=query,
        language=lang,
        intent=result.intent,
        confidence=result.confidence,
        action=action,
        partner_filter=result.partner,
        products=[Product(**p) for p in products],
        clarifying_question=clarifying,
        engine=engine,
        user_context=user_context,
        latency_ms=latency_ms,
    )
