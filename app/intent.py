"""Query understanding: learned classifier + catalog-derived signals.

No hand-coded keyword lists: intent/language/price/vagueness come from the trained
model (app.intent_model), function words are derived from training-corpus document
frequency, partner aliases come from the catalog config, and specificity is judged
against the live index vocabulary. Anything the model can't ground in the catalog
escalates to the LLM (app.llm).
"""

from dataclasses import dataclass
from typing import Optional, Set

from .catalog import PARTNERS
from .intent_model import get_model
from .retrieval import expand_query, fold_token, tokenize

PARTNER_ALIASES = {name: name for name in PARTNERS}  # catalog config, not a lexicon


@dataclass
class IntentResult:
    intent: str             # search | discovery | comparison | customer_support
    language: str           # de | en
    confidence: float       # classifier probability of the predicted intent
    partner: Optional[str]  # navigational target, if any
    is_specific: bool       # query names something retrievable in the joint index
    is_vague: bool          # classifier: too unspecific to search
    price_sensitive: bool   # classifier: price-focused phrasing
    content_tokens: list    # tokens after removing learned function words/partners
    unknown_tokens: list    # content tokens with no match in the index (LLM signal)
    embedding: object = None  # query embedding (reused by the semantic retrieval net)


def detect_language(text: str) -> str:
    return get_model().predict(text)["language"]


def detect(query: str, vocabulary: Set[str]) -> IntentResult:
    model = get_model()
    prediction = model.predict(query)
    tokens = tokenize(query)

    partner = next((PARTNER_ALIASES[t] for t in tokens if t in PARTNER_ALIASES), None)
    content = [t for t in tokens
               if fold_token(t) not in model.stopwords and t not in PARTNER_ALIASES]
    unknown = [t for t in content if not any(f in vocabulary for f in expand_query([t]))]
    is_specific = len(unknown) < len(content)

    return IntentResult(
        intent=prediction["intent"],
        language=prediction["language"],
        confidence=prediction["confidence"],
        partner=partner,
        is_specific=is_specific,
        is_vague=prediction["vague"],
        price_sensitive=prediction["price_sensitive"],
        content_tokens=content,
        unknown_tokens=unknown,
        embedding=prediction.get("embedding"),
    )
