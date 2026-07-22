"""Benchmark transformer embedding models (sentence-transformers) as a semantic tier.

Same decision gate as benchmark_static_embeddings.py, same test suite (imported),
different backends. Candidates:
  - google/embeddinggemma-300m  (308M, multilingual, Matryoshka; gated on HF)
  - mixedbread-ai/mxbai-embed-xsmall-v1  (24M, English-focused)

Usage:
    python scripts/benchmark_transformer_embeddings.py --model <hf-id> [--fallback <hf-id>]
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
from scripts.benchmark_static_embeddings import OFF_TEMPLATE, SEMANTIC_CASES, product_doc  # noqa: E402


def load_model(model_id: str, fallback: str = None):
    from sentence_transformers import SentenceTransformer
    try:
        return model_id, SentenceTransformer(model_id, device="cpu")
    except Exception as exc:
        if fallback:
            print(f"primary model failed ({type(exc).__name__}: {exc}); trying {fallback}")
            return fallback, SentenceTransformer(fallback, device="cpu")
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--fallback", default=None)
    args = parser.parse_args()
    process = psutil.Process()
    catalog = build_catalog()

    rss_before = process.memory_info().rss / 1e6
    t0 = time.perf_counter()
    model_id, model = load_model(args.model, args.fallback)
    load_s = time.perf_counter() - t0
    rss_after = process.memory_info().rss / 1e6

    # EmbeddingGemma uses task prompts; sentence-transformers exposes them via prompt_name.
    # E5 models require literal "query: " / "passage: " prefixes instead.
    prompts = getattr(model, "prompts", {}) or {}
    q_kwargs = {"prompt_name": "query"} if "query" in prompts else {}
    d_kwargs = {"prompt_name": "document"} if "document" in prompts else {}
    is_e5 = "e5" in model_id.lower()

    def encode_q(texts):
        if is_e5:
            texts = [f"query: {t}" for t in texts]
        return model.encode(texts, normalize_embeddings=True, **q_kwargs)

    def encode_d(texts):
        if is_e5:
            texts = [f"passage: {t}" for t in texts]
        return model.encode(texts, normalize_embeddings=True, **d_kwargs)

    print("== footprint ==")
    print(f"model: {model_id}   dim: {model.get_sentence_embedding_dimension()}")
    print(f"load (incl. first download if any): {load_s:.1f}s   RAM delta: {rss_after - rss_before:.0f} MB")

    lat = []
    for query, _ in SEMANTIC_CASES * 3:
        t0 = time.perf_counter()
        encode_q([query])
        lat.append((time.perf_counter() - t0) * 1000)
    lat.sort()
    print(f"\n== encode latency (single query, n={len(lat)}) ==")
    print(f"p50={lat[len(lat)//2]:.1f} ms  p95={lat[int(len(lat)*.95)]:.1f} ms  "
          f"mean={statistics.mean(lat):.1f} ms")

    doc_emb = encode_d([product_doc(p) for p in catalog])

    def top_k(query: str, k: int = 5):
        sims = doc_emb @ encode_q([query])[0]
        idx = np.argsort(-sims)[:k]
        return [catalog[i] for i in idx]

    emb_hits = 0
    print("\n== semantic grounding (top-5 must contain expected product) ==")
    for query, expected in SEMANTIC_CASES:
        names = [p["name"] for p in top_k(query)]
        ok = any(e.lower() in n.lower() for e in expected for n in names)
        emb_hits += ok
        print(f"  [{'OK ' if ok else 'MISS'}] {query!r:42} -> {', '.join(names[:3])}")
    print(f"semantic hit@5: {emb_hits}/{len(SEMANTIC_CASES)}  "
          f"(references: static-emb 15/18, BM25 10/18)")

    from sklearn.linear_model import LogisticRegression
    from app.intent_model import build_training_data, get_model as get_tfidf
    from evaluation.dataset import build_dataset
    rows = build_training_data()
    clf = LogisticRegression(max_iter=1000, C=4.0, random_state=13)
    clf.fit(encode_q([r[0].lower() for r in rows]), [r[1] for r in rows])

    eval_data = build_dataset()
    pred = clf.predict(encode_q([e["query"].lower() for e in eval_data]))
    acc = float(np.mean([p == e["intent"] for p, e in zip(pred, eval_data)]))
    ot_pred = clf.predict(encode_q([q.lower() for q, _ in OFF_TEMPLATE]))
    ot_acc = float(np.mean([p == lbl for p, (_, lbl) in zip(ot_pred, OFF_TEMPLATE)]))
    print(f"\n== intent accuracy ==")
    print(f"eval set (320): {acc:.1%}   off-template (20): {ot_acc:.1%}   "
          f"(references: TF-IDF 97.8% / 70.0%, static-emb 96.9% / 70.0%)")

    from evaluation.harness import score_ranking
    scores = {"hit": [], "mrr": [], "ndcg": []}
    for e in eval_data:
        if not e["relevant_ids"] or e["expected_action"] not in (
                "recommend", "route_to_partner", "compare"):
            continue
        ranked = [p["id"] for p in top_k(e["query"], k=5)]
        s = score_ranking(ranked, set(e["relevant_ids"]))
        for key in scores:
            scores[key].append(s[key])
    print(f"\n== literal-ground-truth retrieval (n={len(scores['hit'])}) ==")
    print(f"this model:     Hit@5={np.mean(scores['hit']):.3f}  MRR@5={np.mean(scores['mrr']):.3f}  "
          f"NDCG@5={np.mean(scores['ndcg']):.3f}")
    print("BM25 (current): Hit@5=1.000  MRR@5=1.000  NDCG@5=0.993")


if __name__ == "__main__":
    main()
