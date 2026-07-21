"""Demo: sends 5 distinct queries (mixed languages & intents) and prints the JSON output.

Usage:
    python demo.py                     # in-process (no server needed)
    python demo.py --url http://localhost:8080   # against a running instance
"""

import argparse
import json
import sys

QUERIES = [
    "Bitte zeige mir Angebote für günstige Windeln",       # de | search (price-aware)
    "I need stuff for a pasta dinner",                      # en | discovery -> theme basket
    "Suche Shampoo bei dm",                                 # de | navigational -> partner route
    "Was ist besser: Bio-Olivenöl oder normales Olivenöl?", # de | comparison
    "Ich habe ein Problem mit meinen PAYBACK Punkten",      # de | customer support
    "Ich brauche ein Geschenk",                             # de | vague gift -> clarifying question
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=None, help="Base URL of a running instance (else in-process)")
    args = parser.parse_args()

    if args.url:
        import httpx
        def ask(query: str) -> dict:
            response = httpx.post(f"{args.url.rstrip('/')}/assist", json={"query": query}, timeout=10)
            response.raise_for_status()
            return response.json()
    else:
        from fastapi.testclient import TestClient
        from app.main import app
        client = TestClient(app)
        client.__enter__()  # trigger lifespan (index build)
        def ask(query: str) -> dict:
            return client.post("/assist", json={"query": query}).json()

    for i, query in enumerate(QUERIES, 1):
        print(f"\n{'=' * 70}\n[{i}] QUERY: {query}\n{'=' * 70}")
        print(json.dumps(ask(query), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.exit(main())
