"""Run a live classifier check against hand-labeled smoke cases."""

from __future__ import annotations

import csv
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from app.services.classifier import classify_voc
from app.services.llm import is_api_configured

CASES_PATH = Path(__file__).with_name("smoke_cases.csv")
REQUIRED_COLUMNS = {"text", "expected_category", "expected_subcategory"}


@dataclass(frozen=True)
class SmokeSummary:
    total: int
    category_hits: int
    subcategory_hits: int
    failed_cases: int

    @property
    def passed(self) -> bool:
        return (
            self.total > 0
            and self.failed_cases == 0
            and self.category_hits == self.total
            and self.subcategory_hits == self.total
        )


def load_cases(path: Path = CASES_PATH) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        fields = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - fields
        if missing:
            raise ValueError(f"Smoke CSV missing columns: {', '.join(sorted(missing))}")
        cases = list(reader)
        if not cases:
            raise ValueError("Smoke CSV does not contain any cases.")
        return cases


def _percentage(hits: int, total: int) -> float:
    return (hits / total * 100) if total else 0.0


def run_cases(
    cases: Sequence[dict[str, str]],
    *,
    classifier: Callable[[str], dict[str, Any]] = classify_voc,
    output: TextIO = sys.stdout,
) -> SmokeSummary:
    category_hits = 0
    subcategory_hits = 0
    failed_cases = 0

    for index, case in enumerate(cases, start=1):
        prediction: dict[str, Any] | None = None
        try:
            prediction = classifier(case["text"])
        except Exception:
            pass

        category_ok = bool(
            prediction
            and prediction.get("category") == case["expected_category"]
        )
        subcategory_ok = bool(
            prediction
            and prediction.get("subcategory") == case["expected_subcategory"]
        )
        case_passed = category_ok and subcategory_ok
        category_hits += int(category_ok)
        subcategory_hits += int(subcategory_ok)
        failed_cases += int(not case_passed)

        predicted_path = (
            f"{prediction['category']} > {prediction['subcategory']}"
            if prediction
            else "(no prediction)"
        )
        print(f"[{index}] {'PASS' if case_passed else 'FAIL'} | {case['text']}", file=output)
        print(
            f"    expected: {case['expected_category']} > {case['expected_subcategory']}",
            file=output,
        )
        print(f"    predicted: {predicted_path}", file=output)
        if prediction:
            print(f"    priority: {prediction['priority']}", file=output)
            print(
                f"    human review: {prediction['requires_human_review']}",
                file=output,
            )
            print(f"    reason: {prediction['reason']}", file=output)
        else:
            print("    error: classification request failed", file=output)
        print(file=output)

    total = len(cases)
    print(
        "Major-category accuracy: "
        f"{category_hits}/{total} ({_percentage(category_hits, total):.1f}%)",
        file=output,
    )
    print(
        "Subcategory accuracy: "
        f"{subcategory_hits}/{total} ({_percentage(subcategory_hits, total):.1f}%)",
        file=output,
    )
    return SmokeSummary(total, category_hits, subcategory_hits, failed_cases)


def main() -> int:
    if not is_api_configured():
        print(
            "OPENAI_API_KEY is not configured. Copy .env.example to .env, "
            "set a real key, and run this command again.",
            file=sys.stderr,
        )
        return 2

    summary = run_cases(load_cases())
    return 0 if summary.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
