"""BigQuery vector search over THREE separate partner tables (challenge: preferred services).

Each partner catalog is ingested into its OWN table (products_dm / products_edeka /
products_amazon) with Vertex AI text embeddings — mirroring how disparate partner
feeds land separately in a warehouse — and a single query fans out with VECTOR_SEARCH
over all three tables simultaneously (UNION ALL), merging by cosine distance.

Usage (requires gcloud auth + bq CLI):
    python scripts/bigquery_vector_search.py --project <gcp-project> "wunder Po Kleinkind"
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.catalog import PARTNERS, build_catalog  # noqa: E402

EMBED_URL = ("https://us-central1-aiplatform.googleapis.com/v1/projects/{project}/locations/"
             "us-central1/publishers/google/models/text-embedding-005:predict")
DATASET = "payback_assistant"
LEGACY_TABLE = "products"  # single-table layout from the first iteration — removed

SCHEMA = [
    {"name": "id", "type": "STRING"}, {"name": "partner", "type": "STRING"},
    {"name": "name", "type": "STRING"}, {"name": "brand", "type": "STRING"},
    {"name": "category", "type": "STRING"}, {"name": "price", "type": "FLOAT64"},
    {"name": "unit", "type": "STRING"},
    {"name": "embedding", "type": "FLOAT64", "mode": "REPEATED"},
]


def _cli(name: str) -> str:
    path = shutil.which(name) or shutil.which(f"{name}.cmd")
    if not path:
        sys.exit(f"{name} CLI not found — install the Google Cloud SDK")
    return path


def _token() -> str:
    return subprocess.check_output([_cli("gcloud"), "auth", "print-access-token"], text=True).strip()


def embed(texts: list, project: str, token: str, task: str) -> list:
    response = httpx.post(
        EMBED_URL.format(project=project),
        headers={"Authorization": f"Bearer {token}"},
        json={"instances": [{"content": t, "task_type": task} for t in texts]},
        timeout=60,
    )
    response.raise_for_status()
    return [p["embeddings"]["values"] for p in response.json()["predictions"]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("query", nargs="?", default="etwas gegen wunden Po bei meinem Kleinkind")
    args = parser.parse_args()
    bq, token = _cli("bq"), _token()

    catalog = build_catalog()
    by_partner = {p: [item for item in catalog if item["partner"] == p] for p in PARTNERS}

    subprocess.run([bq, "--project_id", args.project, "mk", "-d", "--location=EU", DATASET],
                   capture_output=True, text=True)  # idempotent: 'already exists' is fine
    print(f"Removing legacy single table {DATASET}.{LEGACY_TABLE} (if present) ...")
    subprocess.run([bq, "--project_id", args.project, "rm", "-f", "-t", f"{DATASET}.{LEGACY_TABLE}"],
                   capture_output=True, text=True)

    with tempfile.TemporaryDirectory() as tmp:
        schema_path = Path(tmp) / "schema.json"
        schema_path.write_text(json.dumps(SCHEMA), encoding="utf-8")
        for partner, items in by_partner.items():
            table = f"products_{partner}"
            print(f"Embedding + loading {len(items)} products into {DATASET}.{table} ...")
            docs = [f"{p['name']} {p['brand']} {p['category']} {' '.join(p['tags'])}" for p in items]
            vectors = []
            for i in range(0, len(docs), 25):
                vectors.extend(embed(docs[i:i + 25], args.project, token, "RETRIEVAL_DOCUMENT"))
            rows_path = Path(tmp) / f"{table}.jsonl"
            with rows_path.open("w", encoding="utf-8") as fh:
                for product, vector in zip(items, vectors):
                    row = {k: product[k] for k in ("id", "partner", "name", "brand", "category", "price", "unit")}
                    row["embedding"] = vector
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            subprocess.run([bq, "--project_id", args.project, "load", "--replace",
                            "--source_format=NEWLINE_DELIMITED_JSON",
                            f"{DATASET}.{table}", str(rows_path), str(schema_path)],
                           check=True, text=True)

    print(f'Simultaneous VECTOR_SEARCH across all three partner tables for: "{args.query}"')
    query_vector = json.dumps(embed([args.query], args.project, token, "RETRIEVAL_QUERY")[0])
    per_table = "\n      UNION ALL\n".join(
        f"""      SELECT base.partner, base.name, base.category, base.price, distance
      FROM VECTOR_SEARCH(
        TABLE `{DATASET}.products_{partner}`, 'embedding',
        (SELECT {query_vector} AS embedding),
        top_k => 3, distance_type => 'COSINE')"""
        for partner in PARTNERS
    )
    sql = f"""
    WITH hits AS (
{per_table}
    )
    SELECT partner, name, category, price, ROUND(distance, 4) AS distance
    FROM hits ORDER BY distance LIMIT 5
    """
    # SQL goes via stdin: the inlined 768-dim vectors exceed the Windows argv limit.
    subprocess.run([bq, "--project_id", args.project, "query", "--nouse_legacy_sql"],
                   input=sql, check=True, text=True)


if __name__ == "__main__":
    main()
