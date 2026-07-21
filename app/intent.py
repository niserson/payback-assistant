"""Rule-based intent + language detection.

A transparent lexicon/heuristic classifier: deterministic, zero-latency, fully
auditable — the right first rung for a 4-class problem over short retail queries.
(The module boundary is clean, so an LLM/classifier can replace `detect` without
touching the agent.)
"""

import re
from dataclasses import dataclass
from typing import Optional, Set

from .retrieval import CHEAP_TOKENS, expand_query, tokenize

PARTNER_ALIASES = {
    "dm": "dm",
    "edeka": "edeka",
    "amazon": "amazon",
}

_DE_MARKERS = {
    "der", "die", "das", "und", "ich", "mir", "mich", "für", "fuer", "bitte", "zeige", "zeig",
    "brauche", "suche", "mit", "ein", "eine", "einen", "ist", "nicht", "was", "wie", "wo",
    "bei", "von", "günstig", "günstige", "angebote", "etwas", "zutaten", "möchte", "kaufen",
    "habe", "mein", "meine", "meinen", "oder", "besser", "hilfe", "kannst", "du", "gibt", "es",
}
_EN_MARKERS = {
    "the", "i", "need", "for", "a", "an", "is", "show", "me", "please", "what", "how",
    "with", "and", "cheap", "find", "looking", "want", "buy", "some", "stuff", "my",
    "or", "better", "help", "can", "you", "where", "do", "prefer",
}

_SUPPORT = {
    "hilfe", "support", "problem", "probleme", "beschwerde", "reklamation", "rückgabe",
    "rücksendung", "erstattung", "refund", "return", "complaint", "help", "konto",
    "account", "login", "passwort", "password", "hotline", "kundenservice", "punkte",
    "points", "storno", "stornieren", "cancel", "defekt", "broken", "kaputt",
}
_COMPARISON = {
    "vergleich", "vergleiche", "vergleichen", "compare", "comparison", "besser", "better",
    "unterschied", "difference", "vs", "versus", "oder",
}
_DISCOVERY = {
    "stuff", "ideen", "idee", "ideas", "inspiration", "empfehlung", "empfehlungen",
    "empfehle", "recommend", "recommendation", "suggestions", "zutaten", "ingredients",
    "etwas", "something", "geschenk", "gift", "present", "überraschung", "dinner",
    "abendessen", "party", "brunch",
}
# Function words that never indicate a concrete product need.
_STOPWORDS = _DE_MARKERS | _EN_MARKERS | CHEAP_TOKENS | {
    "zeigen", "hallo", "hello", "hi", "danke", "thanks", "gerne", "heute", "morgen",
    "neue", "new", "gut", "good", "auf", "in", "an", "zu", "am", "um",
    "fur", "furs", "fürs", "fuers",  # umlaut-less typing of für
}


@dataclass
class IntentResult:
    intent: str            # search | discovery | comparison | customer_support
    language: str          # de | en
    confidence: float
    partner: Optional[str]  # navigational target, if any
    is_specific: bool       # query names a concrete product/category we can retrieve
    content_tokens: list    # tokens after removing stopwords/partners
    unknown_tokens: list    # content tokens with no match in the index (LLM escalation signal)


def detect_language(text: str) -> str:
    if re.search(r"[äöüßÄÖÜ]", text):
        return "de"
    tokens = set(tokenize(text))
    de_hits = len(tokens & _DE_MARKERS)
    en_hits = len(tokens & _EN_MARKERS)
    if en_hits > de_hits:
        return "en"
    return "de"  # PAYBACK's home market — sensible default for ambiguous queries


def detect(query: str, vocabulary: Set[str]) -> IntentResult:
    tokens = tokenize(query)
    token_set = set(tokens)
    language = detect_language(query)

    partner = next((PARTNER_ALIASES[t] for t in tokens if t in PARTNER_ALIASES), None)

    content = [t for t in tokens if t not in _STOPWORDS and t not in PARTNER_ALIASES]
    # Specific = at least one content token is retrievable in the joint index.
    unknown = [t for t in content if not any(f in vocabulary for f in expand_query([t]))]
    is_specific = len(unknown) < len(content)

    if token_set & _SUPPORT:
        intent, confidence = "customer_support", 0.9
    elif (token_set & _COMPARISON) and is_specific and (token_set & _COMPARISON != {"oder"} or " oder " in f" {query.lower()} "):
        intent, confidence = "comparison", 0.8
    elif token_set & _DISCOVERY:
        intent, confidence = "discovery", 0.8
    elif is_specific or partner:
        intent, confidence = "search", 0.85
    else:
        intent, confidence = "discovery", 0.5  # vague — agent will ask a clarifying question

    return IntentResult(
        intent=intent,
        language=language,
        confidence=confidence,
        partner=partner,
        is_specific=is_specific,
        content_tokens=content,
        unknown_tokens=unknown,
    )
