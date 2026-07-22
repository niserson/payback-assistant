"""Evaluation harness: runs the labeled dataset through the agent and scores it.

Runs the deterministic path in-process (LLM disabled) so results are reproducible
in CI. Metrics:
  - intent / language / action / partner accuracy (+ per-intent breakdown)
  - retrieval quality on examples with ground-truth products: Hit@5, MRR@5, NDCG@5
"""

import math
from collections import defaultdict
from typing import Dict, List

from app.agent import handle
from app.catalog import build_catalog
from app.retrieval import BM25Index

K = 5


def _dcg(gains: List[int]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def score_ranking(ranked_ids: List[str], relevant: set) -> Dict[str, float]:
    top = ranked_ids[:K]
    gains = [1 if pid in relevant else 0 for pid in top]
    hit = 1.0 if any(gains) else 0.0
    mrr = 0.0
    for i, g in enumerate(gains):
        if g:
            mrr = 1.0 / (i + 1)
            break
    ideal = _dcg([1] * min(len(relevant), K))
    ndcg = _dcg(gains) / ideal if ideal else 0.0
    return {"hit": hit, "mrr": mrr, "ndcg": ndcg}


def run(dataset: List[dict], index: BM25Index = None) -> Dict:
    index = index or BM25Index(build_catalog())
    totals = defaultdict(float)
    per_intent = defaultdict(lambda: {"n": 0, "correct": 0})
    confusion = defaultdict(int)
    retrieval = defaultdict(list)

    for i, example in enumerate(dataset):
        response = handle(example["query"], index, max_results=K,
                          user_id=f"eval-{i}", llm_mode="off")
        totals["n"] += 1
        totals["intent_ok"] += response.intent == example["intent"]
        totals["language_ok"] += response.language == example["language"]
        totals["action_ok"] += response.action.type == example["expected_action"]
        per_intent[example["intent"]]["n"] += 1
        per_intent[example["intent"]]["correct"] += response.intent == example["intent"]
        if response.intent != example["intent"]:
            confusion[(example["intent"], response.intent)] += 1
        if example["partner"]:
            totals["nav_n"] += 1
            totals["nav_ok"] += response.partner_filter == example["partner"]
        if example["relevant_ids"] and example["expected_action"] in (
                "recommend", "route_to_partner", "compare"):
            ranked = [p.id for p in response.products]
            scores = score_ranking(ranked, set(example["relevant_ids"]))
            for metric, value in scores.items():
                retrieval[metric].append(value)

    n = totals["n"]
    report = {
        "examples": int(n),
        "intent_accuracy": round(totals["intent_ok"] / n, 4),
        "language_accuracy": round(totals["language_ok"] / n, 4),
        "action_accuracy": round(totals["action_ok"] / n, 4),
        "partner_accuracy": round(totals["nav_ok"] / totals["nav_n"], 4) if totals["nav_n"] else None,
        "retrieval_examples": len(retrieval["hit"]),
        "hit@5": round(sum(retrieval["hit"]) / len(retrieval["hit"]), 4) if retrieval["hit"] else None,
        "mrr@5": round(sum(retrieval["mrr"]) / len(retrieval["mrr"]), 4) if retrieval["mrr"] else None,
        "ndcg@5": round(sum(retrieval["ndcg"]) / len(retrieval["ndcg"]), 4) if retrieval["ndcg"] else None,
        "per_intent": {k: round(v["correct"] / v["n"], 4) for k, v in sorted(per_intent.items())},
        "confusion": {f"{a}->{b}": c for (a, b), c in sorted(confusion.items(), key=lambda x: -x[1])},
    }
    return report


def print_report(report: Dict) -> None:
    print(f"examples            : {report['examples']}")
    print(f"intent accuracy     : {report['intent_accuracy']:.1%}")
    print(f"language accuracy   : {report['language_accuracy']:.1%}")
    print(f"action accuracy     : {report['action_accuracy']:.1%}")
    if report["partner_accuracy"] is not None:
        print(f"partner accuracy    : {report['partner_accuracy']:.1%}")
    print(f"retrieval (n={report['retrieval_examples']}): "
          f"Hit@5={report['hit@5']:.3f}  MRR@5={report['mrr@5']:.3f}  NDCG@5={report['ndcg@5']:.3f}")
    print("per-intent accuracy :", report["per_intent"])
    if report["confusion"]:
        print("top confusions      :", dict(list(report["confusion"].items())[:5]))


if __name__ == "__main__":
    from evaluation.dataset import build_dataset
    print_report(run(build_dataset()))
