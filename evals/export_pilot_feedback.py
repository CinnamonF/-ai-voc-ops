"""Export persisted pilot feedback into a provisional v0.2-style review queue."""

from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path
from urllib import error, parse, request

DEFAULT_TABLE = "pilot_feedback"
_TABLE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"{name}가 필요합니다.")
    return value


def fetch_feedback() -> list[dict]:
    url = _required_env("SUPABASE_URL").rstrip("/")
    key = _required_env("SUPABASE_SERVICE_ROLE_KEY")
    table = os.getenv("SUPABASE_FEEDBACK_TABLE", DEFAULT_TABLE).strip()
    if not _TABLE_RE.fullmatch(table):
        raise RuntimeError("SUPABASE_FEEDBACK_TABLE 이름이 올바르지 않습니다.")

    query = parse.urlencode({"select": "*", "order": "created_at.asc"})
    req = request.Request(
        f"{url}/rest/v1/{table}?{query}",
        headers={
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Accept": "application/json",
        },
    )
    try:
        with request.urlopen(req, timeout=15) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        raise RuntimeError(f"Supabase export 실패 (HTTP {exc.code}).") from exc
    except error.URLError as exc:
        raise RuntimeError("Supabase에 연결하지 못했습니다.") from exc


def to_review_rows(feedback_rows: list[dict]) -> list[dict]:
    rows = []
    for row in feedback_rows:
        feedback_id = str(row.get("feedback_id", "")).strip()
        if not feedback_id:
            continue
        note = str(row.get("feedback_note", "") or "").strip()
        validation = "tester marked prediction correct" if row.get("is_correct") else "tester corrected prediction"
        rows.append(
            {
                "ticket_id": f"PILOT-{feedback_id}",
                "customer_message": row.get("message_redacted", ""),
                "category_gold": row.get("corrected_category", ""),
                "subcategory_gold": row.get("corrected_subcategory", ""),
                "priority_gold": row.get("corrected_priority", ""),
                "sentiment_gold": row.get("corrected_sentiment", ""),
                "human_review_gold": str(bool(row.get("corrected_human_review"))).lower(),
                "label_note": f"{validation}; {note}".strip("; "),
                "dataset_subset": "pilot_feedback",
                "source_type": "pilot_feedback",
                "label_status": "provisional",
                "label_version": "v0.3-pilot",
                "feedback_id": feedback_id,
                "prompt_version": row.get("prompt_version", ""),
                "taxonomy_version": row.get("taxonomy_version", ""),
                "model": row.get("model", ""),
            }
        )
    return rows


def main() -> None:
    feedback = fetch_feedback()
    review_rows = to_review_rows(feedback)

    output = Path("evals/results/pilot_feedback_review_queue.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "ticket_id",
        "customer_message",
        "category_gold",
        "subcategory_gold",
        "priority_gold",
        "sentiment_gold",
        "human_review_gold",
        "label_note",
        "dataset_subset",
        "source_type",
        "label_status",
        "label_version",
        "feedback_id",
        "prompt_version",
        "taxonomy_version",
        "model",
    ]
    with output.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(review_rows)

    print(f"Exported {len(review_rows)} provisional feedback rows -> {output}")
    print("Review them manually before changing label_status to reviewed.")


if __name__ == "__main__":
    main()
