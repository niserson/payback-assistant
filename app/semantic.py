"""Semantic tier: EmbeddingGemma-300m (q4 ONNX) encoder + product embedding matrix.

Adopted after a benchmarked bake-off (see scripts/benchmark_*.py, decision gates in
the repo history): q4 ONNX holds fp32 quality (semantic hit@5 17/18, intent 97.5%,
off-template 90% vs 70% for TF-IDF, literal retrieval 1.000/1.000/0.998) at ~36 ms
single-thread and ~350 MB RAM — torch-free via onnxruntime.

Roles:
  - Query embeddings feed the intent heads (app.intent_model) on EVERY query.
  - Cosine top-k over precomputed product embeddings is the deterministic semantic
    net for retrieval when BM25 has nothing and the LLM is off/unavailable.
    SIM_THRESHOLD calibrated on in-domain vs out-of-domain probes (junk max 0.486,
    true cases min 0.515).

Model weights are downloaded at Docker build time (baked into the image); product
embeddings are precomputed to data/product_emb.npy.
"""

import os
from pathlib import Path
from typing import List, Tuple

import numpy as np

REPO = "onnx-community/embeddinggemma-300m-ONNX"
VARIANT = "model_q4"
QUERY_PREFIX = "task: search result | query: "
DOC_PREFIX = "title: none | text: "
SIM_THRESHOLD = 0.51  # calibrated: true in-domain cases >= 0.515, junk <= 0.502
DOC_EMB_PATH = Path(__file__).resolve().parent.parent / "data" / "product_emb.npy"


class Encoder:
    def __init__(self):
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from transformers import AutoTokenizer
        model_path = hf_hub_download(REPO, f"onnx/{VARIANT}.onnx")
        hf_hub_download(REPO, f"onnx/{VARIANT}.onnx_data")
        self.tokenizer = AutoTokenizer.from_pretrained(REPO)
        options = ort.SessionOptions()
        options.intra_op_num_threads = int(os.getenv("ONNX_THREADS", "2"))
        self.session = ort.InferenceSession(str(model_path), options,
                                            providers=["CPUExecutionProvider"])
        self.output_name = self.session.get_outputs()[0].name
        self.input_names = {i.name for i in self.session.get_inputs()}

    def encode(self, texts: List[str], prefix: str, batch_size: int = 32) -> np.ndarray:
        chunks = []
        for start in range(0, len(texts), batch_size):
            batch = self.tokenizer([prefix + t for t in texts[start:start + batch_size]],
                                   padding=True, truncation=True, max_length=64,
                                   return_tensors="np")
            feeds = {k: v.astype(np.int64) for k, v in batch.items() if k in self.input_names}
            out = self.session.run([self.output_name], feeds)[0]
            if out.ndim == 3:  # token states -> masked mean pooling
                mask = batch["attention_mask"][..., None].astype(np.float32)
                out = (out * mask).sum(1) / mask.sum(1)
            chunks.append(out)
        emb = np.vstack(chunks)
        return emb / np.linalg.norm(emb, axis=1, keepdims=True)


_encoder: Encoder = None
_doc_emb: np.ndarray = None
_products: list = None


def get_encoder() -> Encoder:
    global _encoder
    if _encoder is None:
        _encoder = Encoder()
    return _encoder


def _load_docs():
    global _doc_emb, _products
    if _doc_emb is None:
        from .catalog import build_catalog
        _products = build_catalog()
        if DOC_EMB_PATH.exists():
            _doc_emb = np.load(DOC_EMB_PATH)
        else:
            _doc_emb = build_product_embeddings()
    return _doc_emb, _products


def build_product_embeddings() -> np.ndarray:
    from .catalog import build_catalog
    catalog = build_catalog()
    docs = [f"{p['name']} {p['brand']} {p['category']} {' '.join(p['tags'])}" for p in catalog]
    emb = get_encoder().encode(docs, DOC_PREFIX)
    DOC_EMB_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(DOC_EMB_PATH, emb)
    return emb


def embed_query(query: str) -> np.ndarray:
    return get_encoder().encode([query], QUERY_PREFIX)[0]


def topk(query_vec: np.ndarray, k: int = 5) -> List[Tuple[dict, float]]:
    """Cosine top-k products for a precomputed query embedding."""
    doc_emb, products = _load_docs()
    sims = doc_emb @ query_vec
    order = np.argsort(-sims)[:k]
    return [(products[i], float(sims[i])) for i in order]


def warm() -> None:
    """Load encoder + product matrix eagerly (startup, not first request)."""
    get_encoder()
    _load_docs()


if __name__ == "__main__":
    matrix = build_product_embeddings()
    print(f"encoded {matrix.shape[0]} products (dim {matrix.shape[1]}) -> {DOC_EMB_PATH}")
