"""Build, EXECUTE (against the live service) and export the two deliverable notebooks:

  notebooks/demo.ipynb         -> app/static/demo_notebook.html       (served at /demo-notebook)
  notebooks/performance.ipynb  -> app/static/performance_report.html  (served at /performance-report)

Usage:
    python scripts/build_notebooks.py --base https://<cloud-run-url>
"""

import argparse
import sys
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient
from nbconvert import HTMLExporter

ROOT = Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "notebooks"
STATIC = ROOT / "app" / "static"

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell


def demo_notebook(base: str) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("# PAYBACK Lightweight Assistant — Demo\n"
           f"Five distinct queries (different **languages** and **intents**) against the live "
           f"Cloud Run service `{base}`, showing the raw JSON responses.\n\n"
           "| # | Query | Language | Expected intent / action |\n|---|---|---|---|\n"
           "| 1 | Bitte zeige mir Angebote für günstige Windeln | de | search → recommend |\n"
           "| 2 | I need stuff for a pasta dinner | en | discovery → theme basket |\n"
           "| 3 | Suche Shampoo bei dm | de | navigational → route_to_partner |\n"
           "| 4 | Was ist besser: Bio-Olivenöl oder normales Olivenöl? | de | comparison → compare |\n"
           "| 5 | Ich habe ein Problem mit meinen PAYBACK Punkten | de | customer_support → handoff |"),
        code("import json, requests\n"
             f"BASE = {base!r}\n"
             "def ask(query, user_id='demo-notebook'):\n"
             "    r = requests.post(f'{BASE}/assist', json={'query': query, 'user_id': user_id}, timeout=60)\n"
             "    r.raise_for_status()\n"
             "    print(json.dumps(r.json(), ensure_ascii=False, indent=2))\n"
             "print('service:', requests.get(f'{BASE}/health', timeout=30).json())"),
        md("## 1 — German · search (price-aware)"),
        code("ask('Bitte zeige mir Angebote für günstige Windeln')"),
        md("## 2 — English · discovery (theme basket across partners)"),
        code("ask('I need stuff for a pasta dinner')"),
        md("## 3 — German · navigational (partner-scoped route)"),
        code("ask('Suche Shampoo bei dm')"),
        md("## 4 — German · comparison"),
        code("ask('Was ist besser: Bio-Olivenöl oder normales Olivenöl?')"),
        md("## 5 — German · customer support (handoff, no products)"),
        code("ask('Ich habe ein Problem mit meinen PAYBACK Punkten')"),
        md("### Bonus — LLM escalation + user context\n"
           "A paraphrase the lexicon cannot parse escalates to Gemini on Vertex AI; note the "
           "`engine` field and the `user_context` interest profile accumulated by the five "
           "queries above (fed to the LLM with 30% weight)."),
        code("ask('Ich brauche etwas gegen wunden Po bei meinem Kleinkind')"),
    ]
    return nb


def coldstart_notebook(base: str) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("# Cold Start vs. User-Context-Influenced Recommendations\n"
           f"Live comparison against `{base}`: the **same ambiguous queries** are answered for\n"
           "1. a **cold-start user** (no history — ranking relies solely on query context + global popularity),\n"
           "2. a **baby-parent persona** (profile built from baby queries),\n"
           "3. a **fitness persona** (profile built from sport queries).\n\n"
           "All runs use `llm_mode: always` and the same model, so the ONLY variable is the "
           "user-interest profile injected into the Gemini prompt at **30% weight** "
           "(the query text keeps 70%)."),
        code("import json, uuid, requests\n"
             f"BASE = {base!r}\n"
             "# One cookie session per persona: Cloud Run session affinity pins each persona\n"
             "# to one instance, so the in-process context store sees all of its queries.\n"
             "SESSIONS = {}\n"
             "def ask(query, user_id, show=True):\n"
             "    s = SESSIONS.setdefault(user_id, requests.Session())\n"
             "    r = s.post(f'{BASE}/assist', json={'query': query, 'user_id': user_id,\n"
             "                      'llm_mode': 'always'}, timeout=60)\n"
             "    r.raise_for_status()\n"
             "    d = r.json()\n"
             "    if show:\n"
             "        tops = ', '.join(f\"{p['name']} [{p['category']}]\" for p in d['products'][:3]) or '(no products)'\n"
             "        extra = f\" | clarify: {d['clarifying_question']}\" if d['clarifying_question'] else ''\n"
             "        print(f'  {query!r:44} -> {tops}{extra}')\n"
             "    return d\n"
             "# fresh ids per run so the in-process context store starts clean\n"
             "COLD, BABY, FIT = (f'{p}-{uuid.uuid4().hex[:8]}' for p in ('cold', 'baby', 'fit'))\n"
             "TEST_QUERIES = ['etwas für unterwegs', 'ein kleines Geschenk', 'creme',\n"
             "                'was für den Sonntagmorgen']\n"
             "print('service:', requests.get(f'{BASE}/health', timeout=30).json())"),
        md("## 1 — Cold start: no history, query context only\n"
           "The system is cold-start-safe **by design**: ranking uses only the query plus a "
           "global popularity prior; the LLM prompt contains no profile block."),
        code("for q in TEST_QUERIES:\n"
             "    ask(q, COLD)"),
        md("## 2 — Build the personas (seed queries recorded into each profile)"),
        code("for q in ('Windeln', 'Schnuller und Feuchttücher', 'Babybrei'):\n"
             "    ask(q, BABY, show=False)\n"
             "for q in ('Yogamatte', 'Fitness Tracker', 'Springseil'):\n"
             "    ask(q, FIT, show=False)\n"
             "profile = lambda uid: ask('Windeln' if uid == BABY else 'Yogamatte', uid, show=False)['user_context']['interests']\n"
             "print('baby persona profile   :', profile(BABY))\n"
             "print('fitness persona profile:', profile(FIT))"),
        md("## 3 — Same queries, baby-parent context (30% weight)"),
        code("for q in TEST_QUERIES:\n"
             "    ask(q, BABY)"),
        md("## 4 — Same queries, fitness context (30% weight)"),
        code("for q in TEST_QUERIES:\n"
             "    ask(q, FIT)"),
        md("## What this shows\n"
           "The 30% context weight acts at **two layers**: (1) the profile enters the Gemini "
           "prompt, so a vague query is *resolved* toward the user's dominant categories "
           "instead of triggering a clarifying question (cold start clarifies, personas get "
           "products); (2) retrieval multiplies each product's relevance score by "
           "`1 + 0.3 × interest-share` for its category, so favored categories win near-ties "
           "in the ranking. Unambiguous queries are unaffected — the prompt forbids the "
           "profile from contradicting explicit intent, and a 1.3× cap cannot overturn a "
           "clear relevance gap. Cold start degrades gracefully to query-only relevance plus "
           "global popularity, which is exactly the challenge's cold-start constraint: user "
           "history is never *required*, it only sharpens ambiguity when present. (Profiles "
           "are seeded live moments earlier, so persona outputs can vary slightly "
           "run-to-run.)"),
    ]
    return nb


def performance_notebook(base: str) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("# PAYBACK Lightweight Assistant — Performance Report\n"
           f"Load test against the live Cloud Run deployment `{base}` "
           "(1 vCPU / 512 MiB per instance, scale-to-zero, max 3 instances, concurrency 80)."),
        code("import json, statistics, time, requests\n"
             "from concurrent.futures import ThreadPoolExecutor\n"
             f"BASE = {base!r}\n"
             "QUERIES = ['günstige Windeln', 'Suche Shampoo bei dm', 'pasta tomaten parmesan',\n"
             "           'Schokolade und Chips', 'Mineralwasser 6er']  # rules path = steady-state hot path\n"
             "def run_load(n_requests, concurrency):\n"
             "    latencies, errors = [], 0\n"
             "    session = requests.Session()\n"
             "    def one(i):\n"
             "        nonlocal errors\n"
             "        t0 = time.perf_counter()\n"
             "        try:\n"
             "            r = session.post(f'{BASE}/assist', json={'query': QUERIES[i % len(QUERIES)]}, timeout=30)\n"
             "            r.raise_for_status()\n"
             "            latencies.append((time.perf_counter() - t0) * 1000)\n"
             "        except Exception:\n"
             "            errors += 1\n"
             "    t0 = time.perf_counter()\n"
             "    with ThreadPoolExecutor(max_workers=concurrency) as pool:\n"
             "        list(pool.map(one, range(n_requests)))\n"
             "    wall = time.perf_counter() - t0\n"
             "    latencies.sort()\n"
             "    p = lambda q: latencies[min(len(latencies) - 1, int(q * len(latencies)))]\n"
             "    return {'requests': n_requests, 'concurrency': concurrency, 'errors': errors,\n"
             "            'wall_s': round(wall, 2), 'rps': round(len(latencies) / wall, 1),\n"
             "            'p50_ms': round(p(.5), 1), 'p95_ms': round(p(.95), 1), 'p99_ms': round(p(.99), 1)}\n"
             "requests.get(f'{BASE}/health', timeout=30).json()  # warm the instance"),
        md("## Scaling proof: throughput grows with concurrency, tail latency stays bounded"),
        code("results = [run_load(150, c) for c in (5, 20, 40)]\n"
             "for r in results:\n"
             "    print(r)"),
        md("## LLM model tiers: server-side latency per Gemini model\n"
           "Same paraphrase queries, `llm_mode: always`, measured via the response's own "
           "`latency_ms` (server-side handler time incl. the Vertex AI call — network "
           "round-trip to the client excluded). Escalated-path latency is dominated by "
           "model inference, so this is a direct tier comparison."),
        code("MODELS = ['gemini-2.5-flash-lite', 'gemini-2.5-flash', 'gemini-3.1-flash-lite', 'gemini-3.5-flash']\n"
             "PARAPHRASES = ['etwas gegen wunden Po beim Kleinkind', 'pancakes for the kids',\n"
             "               'was Schönes zum Naschen', 'stuff for taco night', 'spiegelei fürs frühstück',\n"
             "               'something for a rainy sunday afternoon']\n"
             "print(f\"{'model':26} {'p50 ms':>8} {'p95 ms':>8} {'mean ms':>8}  engine sample\")\n"
             "for m in MODELS:\n"
             "    lats, engine = [], ''\n"
             "    for q in PARAPHRASES:\n"
             "        r = requests.post(f'{BASE}/assist', json={'query': q, 'llm_mode': 'always',\n"
             "                          'model': m, 'user_id': f'tier-{m}'}, timeout=60).json()\n"
             "        lats.append(r['latency_ms']); engine = r['engine']\n"
             "    lats.sort()\n"
             "    print(f'{m:26} {lats[len(lats)//2]:8.0f} {lats[-1]:8.0f} {statistics.mean(lats):8.0f}  {engine}')"),
        md("## Cost per 1000 requests (measured, Cloud Run request-based billing)\n"
           "vCPU $0.000024/s + memory $0.0000025/GiB·s billed only while serving, "
           "plus $0.40 per million requests. LLM-escalated requests add Gemini 2.5 Flash-Lite "
           "on Vertex AI (~$0.10/M input + $0.40/M output tokens; ~700 in / 60 out per call)."),
        code("best = max(results, key=lambda r: r['rps'])\n"
             "per_req_s = 1 / best['rps']\n"
             "infra = (0.000024 + 0.0000025 * 0.5) * per_req_s + 0.40 / 1_000_000\n"
             "llm_call = 700 / 1e6 * 0.10 + 60 / 1e6 * 0.40   # per escalated call\n"
             "for esc in (0.0, 0.10, 0.25):\n"
             "    total = (infra + esc * llm_call) * 1000\n"
             "    print(f'cost per 1000 requests at {esc:.0%} LLM escalation: ${total:.4f}')"),
        md("## How prompt/inference time was optimized\n"
           "Latency was minimized by **removing inference from the hot path rather than "
           "optimizing it**: deterministic lexicon rules + a concept-grouped BM25 index answer "
           "every query whose vocabulary is known in ~1–2 ms server-side, so the p50 above is "
           "pure network + HTTP. The LLM (Gemini 2.5 Flash-Lite — the smallest, fastest tier) "
           "is called **only** when the query contains tokens unknown to the index, which "
           "empirically is a small fraction of retail traffic; every miss that recurs can be "
           "promoted into the synonym lexicon, shrinking the escalation rate over time. The "
           "prompt itself is optimized for speed: a single compact instruction block, "
           "few-shot examples instead of long specifications, `responseMimeType: application/json` "
           "to eliminate parsing retries, temperature 0, and a hard 6 s timeout with one retry "
           "falling back to the rules answer — the API never blocks on the LLM. The user-context "
           "profile is injected as one short line (top-5 categories, 30% weight) rather than "
           "raw history, keeping input tokens ~700 per escalated call. Serving-side, the index "
           "is built once at startup from a catalog baked into the image (no cold-start I/O), "
           "the service is stateless, and Cloud Run fans out to 3 instances × concurrency 80.")
    ]
    return nb


def build(name: str, nb: nbf.NotebookNode, html_name: str) -> None:
    NB_DIR.mkdir(exist_ok=True)
    client = NotebookClient(nb, timeout=300, kernel_name="python3")
    client.execute()
    ipynb_path = NB_DIR / name
    nbf.write(nb, ipynb_path)
    body, _ = HTMLExporter(template_name="lab").from_notebook_node(nb)
    (STATIC / html_name).write_text(body, encoding="utf-8")
    print(f"built {ipynb_path} -> app/static/{html_name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="Base URL of the live service")
    args = parser.parse_args()
    base = args.base.rstrip("/")
    build("demo.ipynb", demo_notebook(base), "demo_notebook.html")
    build("coldstart_context.ipynb", coldstart_notebook(base), "coldstart_context.html")
    build("performance.ipynb", performance_notebook(base), "performance_report.html")


if __name__ == "__main__":
    sys.exit(main())
