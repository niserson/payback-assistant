"""Cross-catalog retrieval: pure-Python BM25 over the catalogs' own vocabulary.

Design choice (Occam's razor): the corpus is small and vocabulary-dense, so lexical
BM25 beats an embedding stack on latency, determinism and operational weight — no
model download, sub-millisecond queries, trivially containerizable.

No hand-coded semantics: cross-lingual matching comes from the catalog's own
bilingual tags (data), normalization (umlaut/digraph folding + light stemming) is
algorithmic, and anything outside the catalog vocabulary escalates to the LLM,
which translates it into catalog terms. Cold start is inherent: ranking uses only
the query context plus a global popularity prior (no user history anywhere).
"""

import math
import re
import unicodedata
from collections import Counter
from typing import Dict, List, Optional
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


def fold_token(token: str) -> str:
    """Public canonical form of a single token (folded + stemmed)."""
    return _stem(_fold(token))


def expand_query(tokens: List[str]) -> List[str]:
    """Fold/stem query tokens into their searchable forms."""
    forms = []
    for token in tokens:
        forms.extend(_index_forms(token))
    return forms


def concept_groups(tokens: List[str], stopwords: Optional[set] = None) -> List[frozenset]:
    """One searchable form-set per query CONCEPT (query token).

    Scoring per concept (best matching form) instead of per raw term keeps one
    token with several matching forms from drowning out other concepts. Function
    words (learned from training-corpus document frequency, passed by the caller)
    are not concepts.
    """
    groups = set()
    for token in tokens:
        if stopwords and fold_token(token) in stopwords:
            continue
        groups.add(frozenset(_index_forms(token)))
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
               interests: Optional[Dict[str, float]] = None,
               price_sensitive: bool = False,
               stopwords: Optional[set] = None) -> List[dict]:
        if stopwords is None:
            # Learned function words from the intent model (lazy: avoids an import cycle).
            from .intent_model import get_model
            stopwords = get_model().stopwords
        groups = concept_groups(tokenize(query), stopwords)

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
            if price_sensitive:
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
