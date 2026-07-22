"""Load the synthetic evaluation dataset into its own BigQuery table.

Creates/replaces `payback_assistant.eval_examples` — separate from the three
product tables — so evaluation data lives in the warehouse next to the catalogs.

Usage:
    python scripts/bigquery_load_eval.py --project <gcp-project>
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation.dataset import build_dataset  # noqa: E402

DATASET = "payback_assistant"
TABLE = "eval_examples"

SCHEMA = [
    {"name": "id", "type": "STRING"},
    {"name": "intent", "type": "STRING"},
    {"name": "language", "type": "STRING"},
    {"name": "expected_action", "type": "STRING"},
    {"name": "partner", "type": "STRING"},
    {"name": "query", "type": "STRING"},
    {"name": "relevant_ids", "type": "STRING", "mode": "REPEATED"},
    {"name": "n_relevant", "type": "INT64"},
]


def _cli(name: str) -> str:
    path = shutil.which(name) or shutil.which(f"{name}.cmd")
    if not path:
        sys.exit(f"{name} CLI not found — install the Google Cloud SDK")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    args = parser.parse_args()
    bq = _cli("bq")

    examples = build_dataset()
    with tempfile.TemporaryDirectory() as tmp:
        schema_path = Path(tmp) / "schema.json"
        schema_path.write_text(json.dumps(SCHEMA), encoding="utf-8")
        rows_path = Path(tmp) / "rows.jsonl"
        with rows_path.open("w", encoding="utf-8") as fh:
            for example in examples:
                row = dict(example)
                row["n_relevant"] = len(row["relevant_ids"])
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        subprocess.run([bq, "--project_id", args.project, "load", "--replace",
                        "--source_format=NEWLINE_DELIMITED_JSON",
                        f"{DATASET}.{TABLE}", str(rows_path), str(schema_path)],
                       check=True, text=True)
    print(f"Loaded {len(examples)} examples into {args.project}:{DATASET}.{TABLE}")
    subprocess.run([bq, "--project_id", args.project, "query", "--nouse_legacy_sql"],
                   input=f"SELECT intent, language, COUNT(*) AS n FROM `{DATASET}.{TABLE}` "
                         f"GROUP BY intent, language ORDER BY n DESC",
                   check=True, text=True)


if __name__ == "__main__":
    main()
