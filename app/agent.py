"""Next-Best-Action policy: turns (intent, language, retrieval) into one response.

Decision table per the challenge:
  specific      -> execute cross-catalog search (recommend)
  vague         -> ask a clarifying question
  navigational  -> route to a partner-scoped search
  support       -> hand off to customer service
  comparison    -> retrieve candidates and present a side-by-side
"""

import time
from typing import Optional

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
        "budget": "Welches Budget schwebt dir ungefähr vor?",
    },
    "en": {
        "default": "Could you narrow that down a bit? For example: drugstore items, groceries, or something else — and do you prefer organic brands?",
        "geschenk": "Happy to help! Who is the gift for, and roughly what budget do you have in mind?",
        "budget": "Roughly what budget do you have in mind?",
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


def handle(query: str, index: BM25Index, max_results: int = 5) -> AssistResponse:
    start = time.perf_counter()
    result: IntentResult = detect(query, index.vocabulary())
    lang = result.language

    products: list = []
    action: Action
    clarifying: Optional[str] = None

    if result.intent == "customer_support":
        action = Action(type="support_handoff", detail=_SUPPORT_MSG[lang])

    elif result.intent == "comparison":
        products = index.search(query, top_k=max(2, max_results))
        detail = ("Vergleich der besten Treffer über alle Partner (Preis pro Einheit beachten)."
                  if lang == "de" else
                  "Side-by-side of the top matches across all partners (check price per unit).")
        action = Action(type="compare", detail=detail)

    elif result.partner is not None:
        # Navigational: partner-scoped search; with no residual terms, plain routing.
        products = index.search(query, top_k=max_results, partner=result.partner)
        detail = (f"Weiterleitung zur Partner-Suche: {result.partner}."
                  if lang == "de" else f"Routing to partner-specific search: {result.partner}.")
        action = Action(type="route_to_partner", detail=detail)

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
            products = index.search(query, top_k=max_results)
            action = Action(
                type="recommend",
                detail="Empfehlungen basierend auf deiner Anfrage." if lang == "de"
                else "Recommendations based on your request.",
            )
        else:
            clarifying = _CLARIFY[lang]["default"]
            action = Action(type="clarify", detail=clarifying)

    else:  # search
        products = index.search(query, top_k=max_results)
        if products:
            action = Action(
                type="recommend",
                detail="Suchergebnisse über alle Partner-Kataloge." if lang == "de"
                else "Search results across all partner catalogs.",
            )
        else:
            clarifying = _CLARIFY[lang]["default"]
            action = Action(type="clarify", detail=clarifying)

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
        latency_ms=latency_ms,
    )
