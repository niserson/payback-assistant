"""Benchmark: model2vec static embeddings (potion-multilingual) as a semantic tier.

Decision gate BEFORE any integration. Measures, against the current stack:
  1. Footprint: download size, load time, process RAM delta.
  2. Latency: single-query encode p50/p95 (the would-be hot-path cost).
  3. Semantic grounding: dish/paraphrase queries (the class BM25 cannot match and
     Gemini currently handles) -> does cosine top-5 hit the expected products?
  4. Intent classification: same training data as the TF-IDF model, embeddings +
     LogisticRegression, scored on the 320-example eval set AND an off-template
     paraphrase slice that neither classifier saw in training.
  5. No-regression: harness retrieval metrics (Hit@5/MRR@5/NDCG@5) with the
     embedding retriever on the same strict literal ground truth.

Usage:
    python scripts/benchmark_static_embeddings.py [--model minishlab/potion-multilingual-128M]
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.catalog import build_catalog  # noqa: E402

# ---- semantic grounding cases: (query, substrings that count as a hit in top-5) ----
SEMANTIC_CASES = [
    ("omelette", ["Eier"]),
    ("pochierte eier", ["Eier"]),
    ("zutaten für carbonara", ["Spaghetti", "Parmesan", "Eier"]),
    ("pfannkuchen backen", ["Mehl", "Eier", "Milch"]),
    ("spiegelei", ["Eier"]),
    ("sonnenbrand was hilft", ["After Sun"]),
    ("wunder po beim baby", ["Windeln", "Feuchttücher", "Babybrei", "Schnuller"]),
    ("geschenk für ein kleinkind", ["Lego", "Holzbausteine", "Puzzle"]),
    ("snacks für den filmabend", ["Chips", "Schokolade", "Gummibärchen"]),
    ("brot selber backen", ["Mehl", "Zucker"]),
    ("workout im wohnzimmer", ["Yogamatte", "Springseil", "Fitness"]),
    ("pausenbrot für die schule", ["Toast", "Brot", "Butter", "Gouda"]),
    ("morgens besser wach werden", ["Kaffee", "Milchaufschäumer"]),
    ("zähne weiss bekommen", ["Zahnpasta", "Zahnbürste"]),
    ("trockene haut im winter", ["Bodylotion", "Handcreme", "Gesichtscreme"]),
    ("movie night snacks", ["Chips", "Schokolade", "Gummibärchen"]),
    ("what do i need for pancakes", ["Mehl", "Eier", "Milch"]),
    ("camping am wochenende", ["Campingstuhl", "Rucksack", "Trinkflasche"]),
]

# ---- off-template intent slice: phrasings neither classifier saw in training ----
OFF_TEMPLATE = [
    ("könnten Sie mir eventuell ein paar windeln heraussuchen", "search"),
    ("meine frau schickt mich los um milch zu holen", "search"),
    ("haben Sie zufällig noch shampoo vorrätig", "search"),
    ("i'm after some decent headphones", "search"),
    ("on the hunt for affordable toothpaste", "search"),
    ("wir bräuchten dringend nachschub an kaffee", "search"),
    ("das mit meinen punkten klappt hinten und vorne nicht", "customer_support"),
    ("bei der lieferung ist einiges schiefgelaufen", "customer_support"),
    ("an wen wende ich mich wenn gar nichts mehr funktioniert", "customer_support"),
    ("everything about my last order went wrong", "customer_support"),
    ("who do i talk to about a damaged item", "customer_support"),
    ("lohnt sich das teure olivenöl gegenüber dem billigen überhaupt", "comparison"),
    ("spaghetti oder doch lieber penne, was nehm ich", "comparison"),
    ("is the expensive knife set actually worth it over the cheap one", "comparison"),
    ("mir fehlt noch inspiration fürs wochenendfrühstück", "discovery"),
    ("irgendwelche vorschläge für den grillabend", "discovery"),
    ("got any thoughts for a cozy evening at home", "discovery"),
    ("keine ahnung, überrasch mich einfach", "discovery"),
    ("just show me whatever, honestly", "discovery"),
    ("was würdest du einem gestressten vater empfehlen", "discovery"),
]


def product_doc(p: dict) -> str:
    return f"{p['name']} {p['brand']} {p['category']} {' '.join(p['tags'])}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="minishlab/potion-multilingual-128M")
    args = parser.parse_args()
    process = psutil.Process()
    catalog = build_catalog()

    # ---------- 1. footprint ----------
    rss_before = process.memory_info().rss / 1e6
    t0 = time.perf_counter()
    from model2vec import StaticModel
    model = StaticModel.from_pretrained(args.model)
    load_s = time.perf_counter() - t0
    rss_after = process.memory_info().rss / 1e6
    print(f"== footprint ==")
    print(f"model: {args.model}")
    print(f"load time: {load_s:.1f}s   RAM delta: {rss_after - rss_before:.0f} MB "
          f"(process now {rss_after:.0f} MB)")
    dim = model.encode(["test"]).shape[1]
    print(f"embedding dim: {dim}")

    # ---------- 2. latency ----------
    queries = [c[0] for c in SEMANTIC_CASES] * 6
    lat = []
    for q in queries:
        t0 = time.perf_counter()
        model.encode([q])
        lat.append((time.perf_counter() - t0) * 1000)
    lat.sort()
    print(f"\n== encode latency (single query, n={len(lat)}) ==")
    print(f"p50={lat[len(lat)//2]:.2f} ms  p95={lat[int(len(lat)*.95)]:.2f} ms  "
          f"mean={statistics.mean(lat):.2f} ms")

    # ---------- product embeddings ----------
    docs = [product_doc(p) for p in catalog]
    doc_emb = model.encode(docs)
    doc_emb = doc_emb / np.linalg.norm(doc_emb, axis=1, keepdims=True)

    def top_k(query: str, k: int = 5):
        q = model.encode([query])[0]
        q = q / np.linalg.norm(q)
        sims = doc_emb @ q
        idx = np.argsort(-sims)[:k]
        return [(catalog[i], float(sims[i])) for i in idx]

    # ---------- 3. semantic grounding vs BM25 ----------
    from app.retrieval import BM25Index
    from app.intent_model import get_model as get_intent_model
    index = BM25Index(catalog)
    stop = get_intent_model().stopwords
    emb_hits = bm25_hits = 0
    print(f"\n== semantic grounding (top-5 must contain expected product) ==")
    for query, expected in SEMANTIC_CASES:
        emb_top = [p["name"] for p, _ in top_k(query)]
        bm_top = [h["name"] for h in index.search(query, top_k=5, stopwords=stop)]
        emb_ok = any(e.lower() in n.lower() for e in expected for n in emb_top)
        bm_ok = any(e.lower() in n.lower() for e in expected for n in bm_top)
        emb_hits += emb_ok
        bm25_hits += bm_ok
        mark = "OK " if emb_ok else "MISS"
        print(f"  [{mark}] {query!r:42} emb-> {', '.join(emb_top[:3])}")
    n = len(SEMANTIC_CASES)
    print(f"semantic hit@5: static-embeddings {emb_hits}/{n}  vs  BM25 {bm25_hits}/{n}")

    # ---------- 4. intent: embeddings + LogReg vs TF-IDF ----------
    from sklearn.linear_model import LogisticRegression
    from app.intent_model import build_training_data
    from evaluation.dataset import build_dataset
    rows = build_training_data()
    train_emb = model.encode([r[0].lower() for r in rows])
    clf = LogisticRegression(max_iter=1000, C=4.0, random_state=13)
    clf.fit(train_emb, [r[1] for r in rows])

    eval_data = build_dataset()
    eval_emb = model.encode([e["query"].lower() for e in eval_data])
    emb_pred = clf.predict(eval_emb)
    emb_acc = float(np.mean([p == e["intent"] for p, e in zip(emb_pred, eval_data)]))

    tfidf = get_intent_model()
    tfidf_acc = float(np.mean([tfidf.predict(e["query"])["intent"] == e["intent"]
                               for e in eval_data]))
    print(f"\n== intent accuracy (320-example eval set) ==")
    print(f"TF-IDF+LogReg: {tfidf_acc:.1%}   static-emb+LogReg: {emb_acc:.1%}")

    ot_emb = model.encode([q.lower() for q, _ in OFF_TEMPLATE])
    ot_pred = clf.predict(ot_emb)
    ot_emb_acc = float(np.mean([p == lbl for p, (_, lbl) in zip(ot_pred, OFF_TEMPLATE)]))
    ot_tfidf_acc = float(np.mean([tfidf.predict(q)["intent"] == lbl
                                  for q, lbl in OFF_TEMPLATE]))
    print(f"\n== intent accuracy (off-template paraphrase slice, n={len(OFF_TEMPLATE)}) ==")
    print(f"TF-IDF+LogReg: {ot_tfidf_acc:.1%}   static-emb+LogReg: {ot_emb_acc:.1%}")
    for (q, lbl), tp in zip(OFF_TEMPLATE, ot_pred):
        tf_p = tfidf.predict(q)["intent"]
        if tf_p != lbl or tp != lbl:
            print(f"  {q!r:58} truth={lbl:16} tfidf={tf_p:16} emb={tp}")

    # ---------- 5. no-regression: harness retrieval ground truth ----------
    from evaluation.harness import score_ranking
    scores = {"hit": [], "mrr": [], "ndcg": []}
    for e in eval_data:
        if not e["relevant_ids"] or e["expected_action"] not in (
                "recommend", "route_to_partner", "compare"):
            continue
        ranked = [p["id"] for p, _ in top_k(e["query"], k=5)]
        s = score_ranking(ranked, set(e["relevant_ids"]))
        for key in scores:
            scores[key].append(s[key])
    print(f"\n== literal-ground-truth retrieval (n={len(scores['hit'])}) ==")
    print(f"static-embeddings: Hit@5={np.mean(scores['hit']):.3f}  "
          f"MRR@5={np.mean(scores['mrr']):.3f}  NDCG@5={np.mean(scores['ndcg']):.3f}")
    print(f"BM25 (current):    Hit@5=1.000  MRR@5=1.000  NDCG@5=0.993")


if __name__ == "__main__":
    main()
