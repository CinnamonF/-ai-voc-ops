"""Run a small live classifier check against hand-labeled smoke cases."""

from __future__ import annotations

import csv
from pathlib import Path

from app.services.classifier import classify_voc

CASES_PATH = Path(__file__).with_name("smoke_cases.csv")


def main() -> None:
    with CASES_PATH.open(encoding="utf-8-sig", newline="") as file:
        cases = list(csv.DictReader(file))

    category_hits = 0
    subcategory_hits = 0

    for index, case in enumerate(cases, start=1):
        prediction = classify_voc(case["text"])
        category_ok = prediction["category"] == case["expected_category"]
        subcategory_ok = prediction["subcategory"] == case["expected_subcategory"]
        category_hits += int(category_ok)
        subcategory_hits += int(subcategory_ok)

        print(f"[{index}] {case['text']}")
        print(
            "    expected:",
            f"{case['expected_category']} > {case['expected_subcategory']}",
        )
        print(
            "    predicted:",
            f"{prediction['category']} > {prediction['subcategory']}",
        )
        print(
            "    priority/review:",
            prediction["priority"],
            prediction["requires_human_review"],
        )
        print("    reason:", prediction["reason"])
        print()

    total = len(cases)
    print(f"Major category: {category_hits}/{total}")
    print(f"Subcategory: {subcategory_hits}/{total}")


if __name__ == "__main__":
    main()
