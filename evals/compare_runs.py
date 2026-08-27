"""Compare two evaluation summary JSON files without fabricating experiment results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

METRICS = (
    ("major.accuracy", "Major accuracy"),
    ("major.macro_f1", "Major macro F1"),
    ("subcategory.accuracy", "Subcategory accuracy"),
    ("subcategory.macro_f1", "Subcategory macro F1"),
    ("high_risk.recall", "High-risk recall"),
    ("human_review.precision", "Human-review precision"),
    ("human_review.recall", "Human-review recall"),
)


def get_nested(data: dict, dotted: str) -> float:
    current = data
    for part in dotted.split("."):
        current = current[part]
    return float(current)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two evaluation summaries.")
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))

    if not baseline.get("publishable") or not candidate.get("publishable"):
        print(
            "주의: 둘 중 하나 이상이 provisional 평가입니다. "
            "이 비교 결과를 포트폴리오 성과로 게시하지 마세요."
        )

    print(f"Baseline: {args.baseline}")
    print(f"Candidate: {args.candidate}")
    print()
    for key, label in METRICS:
        before = get_nested(baseline, key)
        after = get_nested(candidate, key)
        delta = after - before
        print(f"{label:24} {before:.3f} -> {after:.3f} ({delta:+.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
