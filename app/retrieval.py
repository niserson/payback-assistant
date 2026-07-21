"""Cross-catalog retrieval: pure-Python BM25 with a German<->English synonym layer.

Design choice (Occam's razor): the corpus is small and vocabulary-dense, so lexical
BM25 + bilingual query expansion beats an embedding stack on latency, determinism
and operational weight — no model download, sub-millisecond queries, trivially
containerizable. Cold start is inherent: ranking uses only the query context plus
a global popularity prior (no user history anywhere).
"""

import math
import re
import unicodedata
from collections import Counter
from typing import Dict, List, Optional

# Bilingual expansion: each group is a set of interchangeable tokens (de <-> en).
_SYNONYM_GROUPS = [
    {"windeln", "windel", "diapers", "diaper"},
    {"feuchttücher", "wipes"},
    {"schnuller", "pacifier"},
    {"shampoo", "haarshampoo"},
    {"zahnpasta", "toothpaste"},
    {"zahnbürste", "toothbrush"},
    {"sonnencreme", "sunscreen"},
    {"waschmittel", "detergent"},
    {"nudeln", "nudel", "pasta", "spaghetti", "penne"},
    {"tomaten", "tomate", "tomatoes", "tomato"},
    {"käse", "cheese", "parmesan"},
    {"olivenöl", "olive", "öl", "oil"},
    {"knoblauch", "garlic"},
    {"zwiebeln", "zwiebel", "onions", "onion"},
    {"eier", "ei", "eggs", "egg"},
    {"milch", "milk"},
    {"brot", "bread"},
    {"äpfel", "apfel", "apples", "apple"},
    {"obst", "fruit", "früchte"},
    {"gemüse", "vegetables"},
    {"fleisch", "meat"},
    {"hähnchen", "chicken"},
    {"hackfleisch", "beef", "bolognese"},
    {"lachs", "salmon", "fisch", "fish"},
    {"kaffee", "coffee"},
    {"frühstück", "breakfast", "müsli", "muesli", "cereal"},
    {"kopfhörer", "headphones", "headphone"},
    {"buch", "bücher", "book", "books", "roman", "novel", "lesen", "reading"},
    {"spielzeug", "toy", "toys"},
    {"geschenk", "gift", "present", "geschenkidee"},
    {"kinder", "kids", "children", "kind"},
    {"baby", "babys"},
    {"küche", "kitchen", "kochen", "cooking"},
    {"pfanne", "pan"},
    {"messer", "knife", "knives"},
    {"lampe", "lamp", "licht", "light"},
    {"wasser", "water"},
    {"bio", "organic", "öko"},
    {"grillen", "bbq", "grill", "barbecue", "bratwurst"},
    {"sport", "fitness"},
    {"joghurt", "yogurt", "yoghurt"},
    {"süßigkeiten", "sweets", "candy", "naschen", "schokolade", "chocolate", "gummibärchen"},
    {"kuchen", "cake", "torte", "gebäck"},
    {"backen", "baking", "backmischung"},
    {"pizza", "tiefkühlpizza"},
    {"avocadobrot", "avocado", "brot"},
    {"mehl", "flour"},
    {"zucker", "sugar"},
    {"butter"},
    {"honig", "honey"},
    {"haut", "skin", "creme", "cream", "lotion"},
]

_SYNONYMS: Dict[str, set] = {}
for group in _SYNONYM_GROUPS:
    for token in group:
        _SYNONYMS.setdefault(token, set()).update(group)

CHEAP_TOKENS = {"günstig", "günstige", "billig", "billige", "cheap", "budget", "angebot", "angebote", "deal", "deals", "offer", "offers", "sale"}

_WORD_RE = re.compile(r"[a-zA-ZäöüÄÖÜß0-9]+")


_DIGRAPH_RE = re.compile(r"(?<!q)ue")  # keep qu- words intact (quelle, quer)


def _fold(text: str) -> str:
    """Lowercase + fold umlauts AND typed-out digraphs to one canonical form.

    'Süßigkeiten', 'Suessigkeiten' and 'Sussigkeiten' all fold to 'sussigkeiten' —
    applied identically to index and query, so occasional collisions stay harmless.
    """
    text = text.lower().replace("ß", "ss")
    text = _DIGRAPH_RE.sub("u", text.replace("ae", "a").replace("oe", "o"))
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


def _stem(token: str) -> str:
    """Very light German/English plural conflation (windeln->windel, apples->apple)."""
    for suffix in ("en", "n", "e", "s"):
        if len(token) > 4 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def tokenize(text: str) -> List[str]:
    return _WORD_RE.findall(text.lower())


def _index_forms(token: str) -> List[str]:
    """All searchable forms of a token: folded + stemmed."""
    folded = _fold(token)
    return list({folded, _stem(folded)})


def expand_query(tokens: List[str]) -> List[str]:
    """Expand query tokens with bilingual synonyms, then fold/stem everything."""
    expanded = []
    for token in tokens:
        expanded.append(token)
        expanded.extend(_SYNONYMS.get(token, ()))
    forms = []
    for token in expanded:
        forms.extend(_index_forms(token))
    return forms


def concept_groups(tokens: List[str]) -> List[frozenset]:
    """One searchable form-set per query CONCEPT (token + its synonyms).

    Scoring per concept (best matching form) instead of per expanded term keeps a
    single concept with many synonyms ("frühstück" -> müsli/cereal/breakfast) from
    drowning out other concepts in the query ("eier").
    """
    groups = set()
    for token in tokens:
        if token in CHEAP_TOKENS:
            continue  # price sensitivity is a ranking modifier, not a concept
        forms = set(_index_forms(token))
        for synonym in _SYNONYMS.get(token, ()):
            forms.update(_index_forms(synonym))
        groups.add(frozenset(forms))
    return list(groups)


class BM25Index:
    """Okapi BM25 over weighted product fields, with popularity/price-aware blending."""

    K1 = 1.5
    B = 0.75
    FIELD_WEIGHTS = {"name": 3, "brand": 2, "tags": 2, "category": 1}

    def __init__(self, products: List[dict]):
        self.products = products
        self.doc_tokens: List[Counter] = []
        self.doc_len: List[int] = []
        self.df: Counter = Counter()
        for product in products:
            bag: Counter = Counter()
            for field, weight in self.FIELD_WEIGHTS.items():
                value = product[field]
                text = " ".join(value) if isinstance(value, list) else str(value)
                for token in tokenize(text):
                    for form in _index_forms(token):
                        bag[form] += weight
            self.doc_tokens.append(bag)
            self.doc_len.append(sum(bag.values()))
            for term in bag:
                self.df[term] += 1
        self.n_docs = len(products)
        self.avg_len = sum(self.doc_len) / max(1, self.n_docs)
        prices = sorted(p["price"] for p in products)
        self._price_rank = {p["id"]: prices.index(p["price"]) / max(1, len(prices) - 1) for p in products}

    def _idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log(1 + (self.n_docs - df + 0.5) / (df + 0.5))

    def _term_score(self, term: str, doc_index: int) -> float:
        tf = self.doc_tokens[doc_index].get(term, 0)
        if not tf:
            return 0.0
        norm = tf * (self.K1 + 1) / (
            tf + self.K1 * (1 - self.B + self.B * self.doc_len[doc_index] / self.avg_len))
        return self._idf(term) * norm

    def search(self, query: str, top_k: int = 5, partner: Optional[str] = None,
               interests: Optional[Dict[str, float]] = None) -> List[dict]:
        raw_tokens = tokenize(query)
        groups = concept_groups(raw_tokens)
        wants_cheap = any(t in CHEAP_TOKENS for t in raw_tokens)

        scored = []
        for i, product in enumerate(self.products):
            if partner and product["partner"] != partner:
                continue
            # Concept-level BM25: each query concept contributes its best form once.
            concept_scores = [max(self._term_score(form, i) for form in group) for group in groups if group]
            score = sum(concept_scores)
            if score <= 0:
                continue
            best_concept = max(range(len(concept_scores)), key=lambda c: concept_scores[c])
            # Cold-start blend: query relevance dominates, global popularity breaks ties.
            final = score * 0.85 + product["popularity"] * 2.0 * 0.15
            if wants_cheap:
                final += (1 - self._price_rank[product["id"]]) * 1.5
            if interests:
                # User-context rank boost: 30% weight, scaled by the user's percentage
                # interest in this product's category (reorders near-ties only).
                final *= 1 + 0.3 * interests.get(product["category"], 0.0) / 100
            scored.append((final, best_concept, product))

        scored.sort(key=lambda triple: -triple[0])
        if scored:
            # Relative cutoff: drop weak tail matches (e.g. brand-only hits) so a
            # dominant exact match isn't diluted by noise.
            top_score = scored[0][0]
            scored = [triple for triple in scored if triple[0] >= 0.25 * top_score]

        # Concept coverage: a shopping list ("Süßigkeiten, Pizza und Kuchen") should
        # surface the best item for EVERY concept before adding runner-ups.
        covered: set = set()
        first_pass, runners = [], []
        for final, concept, product in scored:
            (first_pass if concept not in covered else runners).append((final, product))
            covered.add(concept)
        picked = (first_pass + runners)[:top_k]
        picked.sort(key=lambda pair: -pair[0])

        results = []
        for final, product in picked:
            item = {k: v for k, v in product.items() if k != "popularity"}
            item["score"] = round(final, 3)
            results.append(item)
        return results

    def vocabulary(self) -> set:
        """All indexed terms — used by intent detection to judge query specificity."""
        return set(self.df)
