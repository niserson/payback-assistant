"""Benchmark quantized ONNX EmbeddingGemma against the fp32 reference.

Adoption gate: the int8/q4 artifact must (a) agree with fp32 embeddings
(cosine >= ~0.98), and (b) hold the fp32 quality numbers on the same suite
(semantic 17/18, intent 98.1% / off-template 100%, literal retrieval ~parity).

Usage:
    python scripts/benchmark_onnx_gemma.py [--variant model_quantized|model_q4]
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

REPO = "onnx-community/embeddinggemma-300m-ONNX"
# EmbeddingGemma task prompts (from the model's sentence-transformers config).
QUERY_PREFIX = "task: search result | query: "
DOC_PREFIX = "title: none | text: "


class OnnxGemma:
    def __init__(self, variant: str, threads: int = 4):
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from transformers import AutoTokenizer
        model_path = hf_hub_download(REPO, f"onnx/{variant}.onnx")
        hf_hub_download(REPO, f"onnx/{variant}.onnx_data")
        self.tokenizer = AutoTokenizer.from_pretrained(REPO)
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        self.session = ort.InferenceSession(model_path, options,
                                            providers=["CPUExecutionProvider"])
        self.output_name = self.session.get_outputs()[0].name
        self.input_names = {i.name for i in self.session.get_inputs()}

    def encode(self, texts, prefix: str):
        batch = self.tokenizer([prefix + t for t in texts], padding=True,
                               truncation=True, max_length=256, return_tensors="np")
        feeds = {k: v.astype(np.int64) for k, v in batch.items() if k in self.input_names}
        out = self.session.run([self.output_name], feeds)[0]
        if out.ndim == 3:  # token embeddings -> mean pool over attention mask
            mask = batch["attention_mask"][..., None].astype(np.float32)
            out = (out * mask).sum(1) / mask.sum(1)
        return out / np.linalg.norm(out, axis=1, keepdims=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="model_quantized")
    args = parser.parse_args()
    process = psutil.Process()
    catalog = build_catalog()

    rss0 = process.memory_info().rss / 1e6
    t0 = time.perf_counter()
    model = OnnxGemma(args.variant)
    model.encode(["warmup"], QUERY_PREFIX)
    load_s = time.perf_counter() - t0
    print("== footprint ==")
    print(f"variant: {args.variant}   output: {model.output_name}")
    print(f"load (incl. download if cold): {load_s:.1f}s   "
          f"RAM delta: {process.memory_info().rss / 1e6 - rss0:.0f} MB")

    # ---- agreement with fp32 reference ----
    from sentence_transformers import SentenceTransformer
    ref = SentenceTransformer("unsloth/embeddinggemma-300m", device="cpu")
    probes = [c[0] for c in SEMANTIC_CASES] + [q for q, _ in OFF_TEMPLATE[:6]]
    ref_emb = ref.encode(probes, normalize_embeddings=True, prompt_name="query")
    onnx_emb = model.encode(probes, QUERY_PREFIX)
    cos = np.sum(ref_emb * onnx_emb, axis=1)
    print(f"\n== fp32 agreement (n={len(probes)}) ==")
    print(f"cosine vs fp32: min={cos.min():.4f}  mean={cos.mean():.4f}")

    lat = []
    for q, _ in SEMANTIC_CASES * 3:
        t0 = time.perf_counter()
        model.encode([q], QUERY_PREFIX)
        lat.append((time.perf_counter() - t0) * 1000)
    lat.sort()
    print(f"\n== encode latency (single query, n={len(lat)}) ==")
    print(f"p50={lat[len(lat)//2]:.1f} ms  p95={lat[int(len(lat)*.95)]:.1f} ms")

    doc_emb = model.encode([product_doc(p) for p in catalog], DOC_PREFIX)

    def top_k(query, k=5):
        sims = doc_emb @ model.encode([query], QUERY_PREFIX)[0]
        return [catalog[i] for i in np.argsort(-sims)[:k]]

    hits = 0
    print("\n== semantic grounding ==")
    for query, expected in SEMANTIC_CASES:
        names = [p["name"] for p in top_k(query)]
        ok = any(e.lower() in n.lower() for e in expected for n in names)
        hits += ok
        if not ok:
            print(f"  [MISS] {query!r} -> {', '.join(names[:3])}")
    print(f"semantic hit@5: {hits}/{len(SEMANTIC_CASES)}  (fp32: 17/18)")

    from sklearn.linear_model import LogisticRegression
    from app.intent_model import build_training_data
    from evaluation.dataset import build_dataset
    rows = build_training_data()
    clf = LogisticRegression(max_iter=1000, C=4.0, random_state=13)
    clf.fit(model.encode([r[0].lower() for r in rows], QUERY_PREFIX), [r[1] for r in rows])
    eval_data = build_dataset()
    pred = clf.predict(model.encode([e["query"].lower() for e in eval_data], QUERY_PREFIX))
    acc = float(np.mean([p == e["intent"] for p, e in zip(pred, eval_data)]))
    ot_pred = clf.predict(model.encode([q.lower() for q, _ in OFF_TEMPLATE], QUERY_PREFIX))
    ot_acc = float(np.mean([p == l for p, (_, l) in zip(ot_pred, OFF_TEMPLATE)]))
    print(f"\n== intent accuracy ==")
    print(f"eval set: {acc:.1%} (fp32: 98.1%)   off-template: {ot_acc:.1%} (fp32: 100%)")

    from evaluation.harness import score_ranking
    scores = {"hit": [], "mrr": [], "ndcg": []}
    for e in eval_data:
        if not e["relevant_ids"] or e["expected_action"] not in (
                "recommend", "route_to_partner", "compare"):
            continue
        s = score_ranking([p["id"] for p in top_k(e["query"])], set(e["relevant_ids"]))
        for key in scores:
            scores[key].append(s[key])
    print(f"\n== literal-ground-truth retrieval (n={len(scores['hit'])}) ==")
    print(f"Hit@5={np.mean(scores['hit']):.3f}  MRR@5={np.mean(scores['mrr']):.3f}  "
          f"NDCG@5={np.mean(scores['ndcg']):.3f}  (fp32: 1.000/.993/.993)")


if __name__ == "__main__":
    main()
