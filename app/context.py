"""Per-user query context: rolling interest profile over partner categories.

Every query's resulting categories are recorded per user; the profile (percentage
interest per category) is injected into the LLM call with an explicit 30% weight
(the current query keeps 70% — it must always dominate a stale profile).

Storage is in-process (Occam: correct for a demo, honest about limits — on Cloud
Run each instance has its own memory and scale-to-zero clears it). The module is
a seam: swap `_store` for Firestore/Memorystore for durable multi-instance state.
"""

from collections import Counter, defaultdict
from typing import Dict, List, Optional

CONTEXT_WEIGHT = 0.30  # share of influence granted to the profile in the LLM prompt

_store: Dict[str, Counter] = defaultdict(Counter)


def record(user_id: str, categories: List[str]) -> None:
    """Store this query's category interests for the user."""
    for category in categories:
        _store[user_id][category] += 1


def interests(user_id: str) -> Dict[str, float]:
    """Percentage interest per category, highest first."""
    counts = _store.get(user_id)
    if not counts:
        return {}
    total = sum(counts.values())
    return {cat: round(100 * n / total, 1)
            for cat, n in sorted(counts.items(), key=lambda kv: -kv[1])}


def prompt_context(user_id: str, top_n: int = 5) -> Optional[str]:
    """Compact profile string for the LLM prompt, or None for fresh users."""
    profile = interests(user_id)
    if not profile:
        return None
    top = list(profile.items())[:top_n]
    return ", ".join(f"{cat} {pct}%" for cat, pct in top)


def reset(user_id: Optional[str] = None) -> None:
    if user_id is None:
        _store.clear()
    else:
        _store.pop(user_id, None)
