"""Learned query classifier — replaces all hand-coded intent/language lexicons.

EmbeddingGemma-300m (q4 ONNX, app.semantic) embeddings + logistic-regression
heads:
  intent   search | discovery | comparison | customer_support
  language de | en
  price    price-sensitive query? (drives the low-price ranking boost)
  vague    too unspecific to search? (drives the clarifying-question action)

Feature choice is benchmark-driven (scripts/benchmark_*.py): Gemma embeddings
match TF-IDF on the in-template eval set (97.5% vs 97.8%) but jump from 70% to
90-100% on off-template paraphrases, and the same embedding doubles as the
semantic retrieval net — one ~36 ms encode per query, amortized across both
jobs. Semantics the model cannot ground in the catalog still belong to the
Gemini path.

Training data is generated from templates over the live catalog (tags, names,
categories) — domain knowledge lives in *labeled data*, not in runtime code.
Function words are not hardcoded either: they are derived from document
frequency across the training corpus.
"""

import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from .catalog import PARTNERS, build_catalog
from .retrieval import fold_token, tokenize
from .semantic import QUERY_PREFIX, get_encoder

MODEL_PATH = Path(__file__).resolve().parent.parent / "data" / "intent_gemma.joblib"
_SEED = 13
_STOPWORD_DF = 0.02  # tokens appearing in >=2% of training queries are function words

# ---- labeled training templates (data, not runtime logic) -------------------
_SEARCH_DE = [
    "ich suche {t}", "{t} kaufen", "zeig mir {t}", "wo finde ich {t}", "ich brauche {t}",
    "bitte zeige mir {t}", "haben wir {t} im sortiment", "{t} bestellen", "gibt es {t}",
    "ich möchte {t} kaufen", "ich brauche ein {t}", "gibt es hier ein {t}",
]
_SEARCH_DE_PAIR = ["{t} und {u} kaufen", "ich brauche {t} und {u}", "{t}, {u} und mehr"]
_SEARCH_DE_PRICE = [
    "günstige {t}", "billige {t}", "bitte zeige mir angebote für {t}", "{t} im angebot",
    "wo gibt es {t} am billigsten", "gibt es günstige {t}", "günstige {t} bitte",
]
_SEARCH_EN = [
    "where can i buy {t}", "i need {t}", "show me {t}", "looking for {t}",
    "find {t} for me", "i want to buy {t}", "do you have {t}",
]
_SEARCH_EN_PRICE = ["cheap {t}", "{t} on sale", "best deal on {t}", "budget {t}"]
_DISCOVERY_DE = [
    "zutaten für {th}", "ideen für {th}", "empfehlungen für {th}",
    "was brauche ich für {th}", "inspiration für {th}", "etwas für {th}",
]
_DISCOVERY_EN = [
    "stuff for {th}", "ideas for {th}", "recommendations for {th}",
    "what do i need for {th}", "something for {th}",
]
_COMPARE_DE = [
    "was ist besser: {a} oder {b}?", "vergleiche {a} und {b}", "{a} oder {b}?",
    "unterschied zwischen {a} und {b}", "lohnt sich {a} oder eher {b}",
]
_COMPARE_EN = [
    "{a} vs {b}", "which is better {a} or {b}", "compare {a} and {b}",
    "difference between {a} and {b}",
]
_SUPPORT_DE = [
    "ich habe ein problem mit meinen punkten", "meine erstattung ist nicht angekommen",
    "ich möchte eine beschwerde einreichen", "wie erreiche ich den kundenservice",
    "mein login funktioniert nicht", "ich habe mein passwort vergessen",
    "die lieferung ist defekt angekommen", "ich will meine bestellung stornieren",
    "wo ist meine rückerstattung", "reklamation zu meinem einkauf",
    "hilfe mit meinem konto", "meine payback punkte fehlen", "karte gesperrt was tun",
    "rücksendung anmelden", "hotline nummer bitte", "mein gutschein wurde nicht akzeptiert",
]
_SUPPORT_EN = [
    "my refund has not arrived", "i want to file a complaint", "how do i reach customer support",
    "my login does not work", "i forgot my password", "the delivery arrived broken",
    "i want to cancel my order", "where is my refund", "help me with my account",
    "problem with my points", "my voucher was not accepted", "return an item",
]
_VAGUE_DE = [
    "ich brauche mal was neues", "irgendwas schönes bitte", "hast du eine idee für mich",
    "überrasch mich", "ich weiß nicht was ich will", "einfach mal stöbern",
    "was kannst du mir zeigen", "gibt es was interessantes", "ich brauche ein geschenk",
    "ein kleines geschenk bitte", "geschenkidee gesucht", "was schönes zum verschenken",
    "zeig mir mal irgendwas", "keine ahnung was ich brauche",
    "ich brauche ein geschenk für jemanden", "geschenk für einen freund gesucht",
    "was zum verschenken für jemanden", "ein geschenk für meine kollegin",
]
_VAGUE_EN = [
    "i need something nice", "surprise me please", "any ideas for me",
    "i do not know what i want", "just browsing around", "show me something interesting",
    "i need a gift", "looking for a little present", "no idea what to get",
    "something new maybe", "a gift for someone special", "present for a friend",
]


def build_training_data(seed: int = _SEED) -> List[Tuple[str, str, str, int, int]]:
    """Returns (text, intent, language, price_sensitive, vague) tuples."""
    rng = random.Random(seed)
    catalog = build_catalog()
    tags = sorted({t for p in catalog for t in p["tags"]})
    names = [p["name"] for p in catalog]
    themes = sorted({p["category"].lower() for p in catalog} | {"pasta", "frühstück", "grillen", "baby", "breakfast", "a bbq", "the baby", "a pasta dinner"})
    rows: List[Tuple[str, str, str, int, int]] = []

    for tag in tags:
        for template in rng.sample(_SEARCH_DE, 3):
            rows.append((template.format(t=tag), "search", "de", 0, 0))
        for template in rng.sample(_SEARCH_DE_PRICE, 2):
            rows.append((template.format(t=tag), "search", "de", 1, 0))
        for template in rng.sample(_SEARCH_EN, 2):
            rows.append((template.format(t=tag), "search", "en", 0, 0))
        rows.append((rng.choice(_SEARCH_EN_PRICE).format(t=tag), "search", "en", 1, 0))
    # shopping-list style pairs ("windeln und schnuller kaufen")
    tag_pairs = list(zip(tags, tags[1:] + tags[:1]))
    for a, b in rng.sample(tag_pairs, min(120, len(tag_pairs))):
        rows.append((rng.choice(_SEARCH_DE_PAIR).format(t=a, u=b), "search", "de", 0, 0))
    for theme in themes:
        for template in _DISCOVERY_DE:
            rows.append((template.format(th=theme), "discovery", "de", 0, 0))
        for template in _DISCOVERY_EN:
            rows.append((template.format(th=theme), "discovery", "en", 0, 0))
    shuffled = names[:]
    rng.shuffle(shuffled)
    for a, b in zip(shuffled[::2], shuffled[1::2]):
        for template in rng.sample(_COMPARE_DE, 2):
            rows.append((template.format(a=a, b=b), "comparison", "de", 0, 0))
        rows.append((rng.choice(_COMPARE_EN).format(a=a, b=b), "comparison", "en", 0, 0))
    for text in _SUPPORT_DE:
        rows.append((text, "customer_support", "de", 0, 0))
    for text in _SUPPORT_EN:
        rows.append((text, "customer_support", "en", 0, 0))
    for text in _VAGUE_DE:
        rows.append((text, "discovery", "de", 0, 1))
    for text in _VAGUE_EN:
        rows.append((text, "discovery", "en", 0, 1))
    rng.shuffle(rows)
    return rows


def train(path: Path = MODEL_PATH) -> Dict:
    rows = build_training_data()
    texts = [r[0].lower() for r in rows]
    features = get_encoder().encode(texts, QUERY_PREFIX)

    def fit(labels, balanced=False):
        clf = LogisticRegression(max_iter=1000, C=4.0, random_state=_SEED,
                                 class_weight="balanced" if balanced else None)
        clf.fit(features, labels)
        return clf

    artifact = {
        "intent": fit([r[1] for r in rows]),
        "language": fit([r[2] for r in rows]),
        # binary heads are heavily imbalanced (few positive examples) -> balanced weights
        "price": fit([r[3] for r in rows], balanced=True),
        "vague": fit([r[4] for r in rows], balanced=True),
        "stopwords": _derive_stopwords(texts),
        "n_train": len(rows),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path, compress=3)
    return artifact


def _derive_stopwords(texts: List[str]) -> set:
    """Function words = high document frequency across the training corpus."""
    df: Counter = Counter()
    for text in texts:
        for token in set(fold_token(t) for t in tokenize(text)):
            df[token] += 1
    threshold = _STOPWORD_DF * len(texts)
    return {token for token, count in df.items() if count >= threshold}


class IntentModel:
    def __init__(self, artifact: Dict):
        self._a = artifact
        self.stopwords: set = artifact["stopwords"]

    def predict(self, query: str) -> Dict:
        embedding = get_encoder().encode([query.lower()], QUERY_PREFIX)
        return self.predict_from_embedding(embedding, query)

    def predict_from_embedding(self, embedding: np.ndarray, query: str) -> Dict:
        features = embedding.reshape(1, -1) if embedding.ndim == 1 else embedding
        probs = self._a["intent"].predict_proba(features)[0]
        best = probs.argmax()
        return {
            "intent": self._a["intent"].classes_[best],
            "confidence": round(float(probs[best]), 3),
            "language": self._a["language"].predict(features)[0],
            "price_sensitive": bool(self._a["price"].predict(features)[0]),
            "vague": bool(self._a["vague"].predict(features)[0]),
            "embedding": features[0],
        }


_model: IntentModel = None


def get_model() -> IntentModel:
    global _model
    if _model is None:
        artifact = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else train()
        _model = IntentModel(artifact)
    return _model


if __name__ == "__main__":
    art = train()
    print(f"trained on {art['n_train']} examples -> {MODEL_PATH}")
    print(f"learned {len(art['stopwords'])} function words, e.g. "
          f"{sorted(list(art['stopwords']))[:12]}")
