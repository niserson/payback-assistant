"""BigQuery vector search over the partner catalogs (challenge: preferred services).

Ingests all three partner catalogs into BigQuery with Vertex AI text embeddings and
runs a semantic VECTOR_SEARCH — the production-scale retrieval path (the serving API
keeps in-memory BM25, which is faster and cheaper at demo catalog size).

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
from app.catalog import build_catalog  # noqa: E402

EMBED_URL = ("https://us-central1-aiplatform.googleapis.com/v1/projects/{project}/locations/"
             "us-central1/publishers/google/models/text-embedding-005:predict")
DATASET = "payback_assistant"
TABLE = "products"

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
    print(f"Embedding {len(catalog)} products via Vertex AI text-embedding-005 ...")
    docs = [f"{p['name']} {p['brand']} {p['category']} {' '.join(p['tags'])}" for p in catalog]
    vectors = []
    for i in range(0, len(docs), 25):
        vectors.extend(embed(docs[i:i + 25], args.project, token, "RETRIEVAL_DOCUMENT"))

    with tempfile.TemporaryDirectory() as tmp:
        rows_path = Path(tmp) / "rows.jsonl"
        schema_path = Path(tmp) / "schema.json"
        schema_path.write_text(json.dumps(SCHEMA), encoding="utf-8")
        with rows_path.open("w", encoding="utf-8") as fh:
            for product, vector in zip(catalog, vectors):
                row = {k: product[k] for k in ("id", "partner", "name", "brand", "category", "price", "unit")}
                row["embedding"] = vector
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

        print(f"Loading into BigQuery {args.project}:{DATASET}.{TABLE} ...")
        subprocess.run([bq, "--project_id", args.project, "mk", "-d", "--location=EU", DATASET],
                       capture_output=True, text=True)  # idempotent: 'already exists' is fine
        subprocess.run([bq, "--project_id", args.project, "load", "--replace",
                        "--source_format=NEWLINE_DELIMITED_JSON",
                        f"{DATASET}.{TABLE}", str(rows_path), str(schema_path)],
                       check=True, text=True)

    print(f'Semantic VECTOR_SEARCH for: "{args.query}"')
    query_vector = embed([args.query], args.project, token, "RETRIEVAL_QUERY")[0]
    sql = f"""
    SELECT base.partner, base.name, base.price, ROUND(distance, 4) AS distance
    FROM VECTOR_SEARCH(
      TABLE `{DATASET}.{TABLE}`, 'embedding',
      (SELECT {json.dumps(query_vector)} AS embedding),
      top_k => 5, distance_type => 'COSINE')
    ORDER BY distance
    """
    # SQL goes via stdin: the inlined 768-dim vector exceeds the Windows argv limit.
    subprocess.run([bq, "--project_id", args.project, "query", "--nouse_legacy_sql"],
                   input=sql, check=True, text=True)


if __name__ == "__main__":
    main()
