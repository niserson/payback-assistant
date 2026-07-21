"""Next-Best-Action policy: turns (intent, language, retrieval) into one response.

Decision table per the challenge:
  specific      -> execute cross-catalog search (recommend)
  vague         -> ask a clarifying question
  navigational  -> route to a partner-scoped search
  support       -> hand off to customer service
  comparison    -> retrieve candidates and present a side-by-side

Hybrid understanding: deterministic rules first (~0.2 ms). Only when rules find no
retrievable product term does the agent consult Gemini (app.llm) to paraphrase the
need into German catalog keywords or a clarifying question — with silent fallback
to the rule path if the LLM is unavailable.
"""

import time
from typing import Optional

from . import context, llm
from .intent import IntentResult, detect
from .retrieval import BM25Index
from .schemas import Action, AssistResponse, Product

# Theme baskets let discovery queries ("stuff for a pasta dinner") expand into
# concrete retrievable needs without any user history (cold-start friendly).
_THEMES = {
    "pasta": "spaghetti passierte tomaten parmesan olivenöl basilikum knoblauch hackfleisch",
    "spaghetti": "spaghetti passierte tomaten parmesan olivenöl basilikum knoblauch",
    "dinner": "pasta tomaten parmesan olivenöl",
    "abendessen": "pasta tomaten parmesan olivenöl",
    "frühstück": "müsli milch honig brot butter eier orangensaft",
    "breakfast": "müsli milch honig brot butter eier orangensaft",
    "grillen": "bratwurst grillwürstchen campingstuhl",
    "bbq": "bratwurst grillwürstchen campingstuhl",
    "baby": "windeln feuchttücher babybrei schnuller holzbausteine",
    "party": "mineralwasser orangensaft bratwurst brettspiel",
}

_CLARIFY = {
    "de": {
        "default": "Kannst du das etwas eingrenzen? Suchst du z.B. Drogerie-Artikel, Lebensmittel oder etwas anderes — und bevorzugst du Bio-Marken?",
        "geschenk": "Gerne! Für wen ist das Geschenk gedacht und welches Budget hast du ungefähr?",
    },
    "en": {
        "default": "Could you narrow that down a bit? For example: drugstore items, groceries, or something else — and do you prefer organic brands?",
        "geschenk": "Happy to help! Who is the gift for, and roughly what budget do you have in mind?",
    },
}

_SUPPORT_MSG = {
    "de": "Das klingt nach einem Anliegen für unseren Kundenservice. Ich leite dich weiter: PAYBACK Service unter 089 / 996 331 60 oder im Hilfe-Center der App (Konto → Hilfe).",
    "en": "That sounds like a case for our customer service. Routing you now: PAYBACK Service at +49 89 996 331 60 or via the in-app Help Center (Account → Help).",
}


def _theme_query(tokens: list) -> Optional[str]:
    for token in tokens:
        if token in _THEMES:
            return _THEMES[token]
    return None


def handle(query: str, index: BM25Index, max_results: int = 5, user_id: str = "anon") -> AssistResponse:
    start = time.perf_counter()
    result: IntentResult = detect(query, index.vocabulary())
    lang = result.language
    engine = "rules"
    search_query = query
    llm_clarify: Optional[str] = None

    # Escalate to the LLM when the fast path can't fully parse the query: either
    # nothing is retrievable, or some content tokens are unknown to the index
    # (e.g. "spiegelei fur fruhstuck" — 'frühstück' matches but 'spiegelei' doesn't,
    # so rules alone would return generic breakfast items instead of eggs).
    needs_llm = (not result.is_specific) or bool(result.unknown_tokens)
    if llm.available() and needs_llm and result.intent in ("search", "discovery"):
        understood = llm.classify(query, user_context=context.prompt_context(user_id))
        if understood:
            engine = f"rules+{llm.model_name()}@{llm.backend()}"
            lang = understood["language"]
            result.intent = understood["intent"]
            result.partner = understood["partner"] or result.partner
            if understood["search_terms"]:
                search_query = understood["search_terms"]
                result.is_specific = True
            elif understood["clarifying_question"]:
                llm_clarify = understood["clarifying_question"]

    products: list = []
    action: Action
    clarifying: Optional[str] = None

    if result.intent == "customer_support":
        action = Action(type="support_handoff", detail=_SUPPORT_MSG[lang])

    elif result.intent == "comparison":
        products = index.search(search_query, top_k=max(2, max_results))
        detail = ("Vergleich der besten Treffer über alle Partner (Preis pro Einheit beachten)."
                  if lang == "de" else
                  "Side-by-side of the top matches across all partners (check price per unit).")
        action = Action(type="compare", detail=detail)

    elif result.partner is not None:
        # Navigational: partner-scoped search; with no residual terms, plain routing.
        products = index.search(search_query, top_k=max_results, partner=result.partner)
        detail = (f"Weiterleitung zur Partner-Suche: {result.partner}."
                  if lang == "de" else f"Routing to partner-specific search: {result.partner}.")
        action = Action(type="route_to_partner", detail=detail)

    elif llm_clarify:
        clarifying = llm_clarify
        action = Action(type="clarify", detail=clarifying)

    elif result.intent == "discovery":
        theme = _theme_query(result.content_tokens)
        # Gifts are inherently vague without recipient/budget -> clarify per the challenge.
        if any(t in ("geschenk", "gift", "present", "geschenkidee") for t in result.content_tokens):
            clarifying = _CLARIFY[lang]["geschenk"]
            action = Action(type="clarify", detail=clarifying)
        elif theme:
            products = index.search(theme, top_k=max_results)
            detail = ("Themen-Warenkorb passend zu deiner Anfrage, partnerübergreifend zusammengestellt."
                      if lang == "de" else
                      "Theme basket assembled for your request across all partner catalogs.")
            action = Action(type="recommend", detail=detail)
        elif result.is_specific:
            products = index.search(search_query, top_k=max_results)
            action = Action(
                type="recommend",
                detail="Empfehlungen basierend auf deiner Anfrage." if lang == "de"
                else "Recommendations based on your request.",
            )
        else:
            clarifying = _CLARIFY[lang]["default"]
            action = Action(type="clarify", detail=clarifying)

    else:  # search
        products = index.search(search_query, top_k=max_results)
        if products:
            action = Action(
                type="recommend",
                detail="Suchergebnisse über alle Partner-Kataloge." if lang == "de"
                else "Search results across all partner catalogs.",
            )
        else:
            clarifying = _CLARIFY[lang]["default"]
            action = Action(type="clarify", detail=clarifying)

    # Store this query's interests, then report the updated profile back.
    context.record(user_id, [p["category"] for p in products])
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
