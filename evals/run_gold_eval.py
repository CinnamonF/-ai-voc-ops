"""Run the versioned VOC evaluation dataset through the live classifier."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from app.services.batch import analyze_batch
from app.services.evaluation import EvaluationDataError, evaluate_results, summary_dict
from app.services.llm import is_api_configured

DEFAULT_DATASET = Path("evals/gold/voc_gold_seed_v0.2.csv")
DEFAULT_OUTPUT_DIR = Path("evals/results")
PROMPT_VERSION = "v0.1"
TAXONOMY_VERSION = "v0.1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VOC gold evaluation.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--include-provisional",
        action="store_true",
        help="Calculate exploratory metrics from provisional synthetic labels. These metrics are not publishable.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not is_api_configured():
        print("OPENAI_API_KEY가 설정되어 있지 않습니다.")
        return 2
    if not args.dataset.exists():
        print(f"평가 데이터셋을 찾을 수 없습니다: {args.dataset}")
        return 2

    gold = pd.read_csv(args.dataset)
    results = analyze_batch(gold, "customer_message")
    results["prompt_version"] = PROMPT_VERSION
    results["taxonomy_version"] = TAXONOMY_VERSION

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"gold_eval_{timestamp}"

    predictions_path = args.output_dir / f"{stem}_predictions.csv"
    results.to_csv(predictions_path, index=False, encoding="utf-8-sig")
    print(f"Predictions: {predictions_path}")

    try:
        report = evaluate_results(results, include_provisional=args.include_provisional)
    except EvaluationDataError as exc:
        print(f"평가 수치를 계산하지 않았습니다: {exc}")
        print(
            "현재 synthetic seed는 provisional 상태입니다. 사람이 검수한 뒤 "
            "label_status=reviewed로 변경하거나, 개발 확인용으로만 --include-provisional을 사용하세요."
        )
        return 0

    summary_path = args.output_dir / f"{stem}_summary.json"
    summary_path.write_text(
        json.dumps(summary_dict(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report.major_confusion.to_csv(
        args.output_dir / f"{stem}_major_confusion.csv", encoding="utf-8-sig"
    )
    report.subcategory_confusion.to_csv(
        args.output_dir / f"{stem}_subcategory_confusion.csv", encoding="utf-8-sig"
    )
    report.per_subcategory.to_csv(
        args.output_dir / f"{stem}_per_subcategory.csv", index=False, encoding="utf-8-sig"
    )
    report.errors.to_csv(
        args.output_dir / f"{stem}_errors.csv", index=False, encoding="utf-8-sig"
    )

    prefix = "PUBLISHABLE" if report.publishable else "EXPLORATORY / PROVISIONAL"
    print(f"[{prefix}] rows evaluated: {report.rows_evaluated}")
    print(f"Major accuracy: {report.major['accuracy']:.3f}")
    print(f"Major macro F1: {report.major['macro_f1']:.3f}")
    print(f"Subcategory accuracy: {report.subcategory['accuracy']:.3f}")
    print(f"Subcategory macro F1: {report.subcategory['macro_f1']:.3f}")
    print(f"High-risk recall: {report.high_risk['recall']:.3f}")
    print(f"Human-review precision: {report.human_review['precision']:.3f}")
    print(f"Human-review recall: {report.human_review['recall']:.3f}")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
