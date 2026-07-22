"""Synthetic labeled evaluation dataset (>= 300 examples), generated deterministically
from the product catalog via templates.

Ground-truth relevance is deliberately INDEPENDENT of the retriever: a product is
relevant to a query term iff the term literally appears in its tag list or name
(no synonyms, no folding, no stemming). The retriever earns its metrics by bridging
umlauts, digraphs, plurals and DE<->EN synonyms on top of that strict ground truth.

Labels per example:
  intent          search | discovery | comparison | customer_support
  language        de | en
  expected_action recommend | route_to_partner | compare | support_handoff | clarify
  partner         expected navigational target (or None)
  relevant_ids    product ids that count as correct retrievals (may be empty)
"""

import random
from typing import Dict, List

from app.catalog import build_catalog

_SEARCH_DE = ["ich suche {t}", "{t} kaufen", "zeig mir {t}", "wo finde ich {t}", "günstige {t}"]
_SEARCH_EN = ["where can I buy {t}", "I need {t}", "cheap {t}", "show me some {t}"]
# English tags that exist verbatim in the catalog (validated at build time).
_EN_TAGS = [
    "diapers", "wipes", "pacifier", "toothpaste", "toothbrush", "sunscreen", "detergent",
    "pasta", "tomatoes", "cheese", "garlic", "onions", "eggs", "milk", "bread", "apples",
    "bananas", "chicken", "salmon", "coffee", "yogurt", "honey", "chocolate", "sweets",
    "candy", "pizza", "avocado", "toast", "flour", "sugar", "cake", "headphones", "cable",
    "battery", "book", "novel", "puzzle", "fitness", "tracker", "backpack", "humidifier",
    "kettle", "pan", "knife", "lamp", "light", "gift",
]
_DISCOVERY_TEMPLATES = [
    ("de", "zutaten für {theme}"), ("de", "ideen für {theme}"),
    ("de", "empfehlungen für {theme}"), ("en", "stuff for {theme_en}"),
]
_THEME_TAGS = {  # theme -> (de word, en phrase, ground-truth tags)
    "pasta": ("pasta", "a pasta dinner", ["pasta"]),
    "frühstück": ("frühstück", "breakfast", ["frühstück", "breakfast"]),
    "grillen": ("grillen", "a bbq", ["grillen", "bbq"]),
    "baby": ("baby", "the baby", ["baby"]),
}
_SUPPORT = [
    ("de", "ich habe ein problem mit meinen punkten"),
    ("de", "meine erstattung ist nicht angekommen"),
    ("de", "ich möchte eine beschwerde einreichen"),
    ("de", "wie erreiche ich den kundenservice"),
    ("de", "mein login funktioniert nicht"),
    ("de", "ich habe mein passwort vergessen"),
    ("de", "die lieferung ist defekt angekommen"),
    ("de", "ich will meine bestellung stornieren"),
    ("de", "wo ist meine rückerstattung"),
    ("de", "reklamation zu meinem einkauf"),
    ("en", "my refund has not arrived"),
    ("en", "i want to file a complaint"),
    ("en", "how do i reach customer support"),
    ("en", "my login does not work"),
    ("en", "i forgot my password"),
    ("en", "the delivery arrived broken"),
    ("en", "i want to cancel my order"),
    ("en", "where is my refund"),
    ("en", "help me with my account"),
    ("en", "problem with my points"),
]
_VAGUE = [
    ("de", "ich brauche mal was neues"), ("de", "irgendwas schönes bitte"),
    ("de", "hast du eine idee für mich"), ("de", "überrasch mich"),
    ("de", "ich weiß nicht was ich will"), ("de", "einfach mal stöbern"),
    ("de", "was kannst du mir zeigen"), ("de", "gibt es was interessantes"),
    ("de", "ich brauche ein geschenk"), ("de", "ein kleines geschenk bitte"),
    ("en", "i need something nice"), ("en", "surprise me please"),
    ("en", "any ideas for me"), ("en", "i do not know what i want"),
    ("en", "just browsing around"), ("en", "show me something interesting"),
    ("en", "i need a gift"), ("en", "looking for a little present"),
]
_NAV_DE = ["{t} bei {p}", "suche {t} bei {p}", "zeig mir {t} von {p}"]


def _relevant(catalog: List[dict], term: str, partner: str = None) -> List[str]:
    """Strict ground truth: literal tag membership or name substring, no NLP."""
    term = term.lower()
    hits = []
    for product in catalog:
        if partner and product["partner"] != partner:
            continue
        if term in product["tags"] or term in product["name"].lower():
            hits.append(product["id"])
    return hits


def build_dataset(seed: int = 7) -> List[Dict]:
    rng = random.Random(seed)
    catalog = build_catalog()
    examples: List[Dict] = []

    def add(intent, language, action, query, relevant=None, partner=None):
        examples.append({
            "id": f"ex-{len(examples) + 1:03d}", "intent": intent, "language": language,
            "expected_action": action, "query": query,
            "relevant_ids": relevant or [], "partner": partner,
        })

    # --- search (German): one query per product from its primary tag ---
    for product in catalog:
        tag = product["tags"][0]
        template = rng.choice(_SEARCH_DE)
        add("search", "de", "recommend", template.format(t=tag), _relevant(catalog, tag))

    # --- search (English): curated English tags present in the catalog, 2 phrasings each ---
    for tag in _EN_TAGS:
        relevant = _relevant(catalog, tag)
        assert relevant, f"eval dataset bug: english tag {tag!r} matches no product"
        for template in rng.sample(_SEARCH_EN, 2):
            add("search", "en", "recommend", template.format(t=tag), relevant)

    # --- discovery: theme queries with explicit discovery markers ---
    for theme, (de_word, en_phrase, truth_tags) in _THEME_TAGS.items():
        relevant = sorted({pid for t in truth_tags for pid in _relevant(catalog, t)})
        for language, template in _DISCOVERY_TEMPLATES:
            query = template.format(theme=de_word, theme_en=en_phrase)
            add("discovery", language, "recommend", query, relevant)

    # --- comparison: pairs of products from the same category ---
    by_category: Dict[str, List[dict]] = {}
    for product in catalog:
        by_category.setdefault(product["category"], []).append(product)
    pairs = [(items[0], items[1]) for items in by_category.values() if len(items) >= 2]
    for a, b in pairs[:30]:
        add("comparison", "de", "compare", f"was ist besser: {a['name']} oder {b['name']}?",
            [a["id"], b["id"]])

    # --- customer support ---
    for language, query in _SUPPORT:
        add("customer_support", language, "support_handoff", query)

    # --- navigational: partner-scoped searches ---
    for partner in ("dm", "edeka", "amazon"):
        partner_products = [p for p in catalog if p["partner"] == partner]
        for product in rng.sample(partner_products, 15):
            tag = product["tags"][0]
            template = rng.choice(_NAV_DE)
            add("search", "de", "route_to_partner", template.format(t=tag, p=partner),
                _relevant(catalog, tag, partner=partner), partner=partner)

    # --- vague: must ask a clarifying question ---
    for language, query in _VAGUE:
        add("discovery", language, "clarify", query)

    return examples


if __name__ == "__main__":
    data = build_dataset()
    from collections import Counter
    print(len(data), "examples")
    print(Counter(e["intent"] for e in data))
    print(Counter(e["language"] for e in data))
    print(Counter(e["expected_action"] for e in data))
