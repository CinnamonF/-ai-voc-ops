"""Evaluation utilities for versioned VOC classifier experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

GOLD_COLUMNS = (
    "ticket_id",
    "customer_message",
    "category_gold",
    "subcategory_gold",
    "priority_gold",
    "sentiment_gold",
    "human_review_gold",
    "label_status",
    "label_version",
)

PREDICTION_COLUMNS = (
    "category",
    "subcategory",
    "priority",
    "sentiment",
    "requires_human_review",
    "analysis_status",
)

PUBLISHABLE_LABEL_STATUS = "reviewed"


class EvaluationDataError(ValueError):
    """Raised when an evaluation dataset cannot support a valid comparison."""


@dataclass(frozen=True)
class EvaluationReport:
    rows_total: int
    rows_evaluated: int
    rows_failed: int
    publishable: bool
    major: dict[str, float]
    subcategory: dict[str, float]
    high_risk: dict[str, float]
    human_review: dict[str, float]
    usage: dict[str, Any]
    versions: dict[str, str]
    major_confusion: pd.DataFrame
    subcategory_confusion: pd.DataFrame
    errors: pd.DataFrame
    per_subcategory: pd.DataFrame


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise EvaluationDataError(
            "평가에 필요한 컬럼이 없습니다: " + ", ".join(missing)
        )


def _as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin(
        {"true", "1", "yes", "y"}
    )


def _safe_divide(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _classification_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> tuple[dict[str, float], pd.DataFrame]:
    labels = sorted(set(y_true.dropna().astype(str)) | set(y_pred.dropna().astype(str)))
    rows: list[dict[str, Any]] = []
    for label in labels:
        actual = y_true.astype(str).eq(label)
        predicted = y_pred.astype(str).eq(label)
        tp = int((actual & predicted).sum())
        fp = int((~actual & predicted).sum())
        fn = int((actual & ~predicted).sum())
        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, tp + fn)
        f1 = _safe_divide(2 * precision * recall, precision + recall)
        rows.append(
            {
                "label": label,
                "support": int(actual.sum()),
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    per_label = pd.DataFrame(rows)
    accuracy = float((y_true.astype(str) == y_pred.astype(str)).mean()) if len(y_true) else 0.0
    metrics = {
        "accuracy": accuracy,
        "macro_precision": float(per_label["precision"].mean()) if not per_label.empty else 0.0,
        "macro_recall": float(per_label["recall"].mean()) if not per_label.empty else 0.0,
        "macro_f1": float(per_label["f1"].mean()) if not per_label.empty else 0.0,
    }
    return metrics, per_label


def _binary_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    truth = _as_bool(y_true)
    pred = _as_bool(y_pred)
    tp = int((truth & pred).sum())
    fp = int((~truth & pred).sum())
    fn = int((truth & ~pred).sum())
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    return {
        "precision": precision,
        "recall": recall,
        "f1": _safe_divide(2 * precision * recall, precision + recall),
        "support": int(truth.sum()),
    }


def _high_risk_metrics(frame: pd.DataFrame) -> dict[str, float]:
    gold = frame["priority_gold"].astype(str).str.lower().isin({"high", "critical"})
    pred = frame["priority"].astype(str).str.lower().isin({"high", "critical"})
    tp = int((gold & pred).sum())
    fn = int((gold & ~pred).sum())
    fp = int((~gold & pred).sum())
    return {
        "precision": _safe_divide(tp, tp + fp),
        "recall": _safe_divide(tp, tp + fn),
        "support": int(gold.sum()),
    }


def _confusion(frame: pd.DataFrame, gold_column: str, pred_column: str) -> pd.DataFrame:
    return pd.crosstab(
        frame[gold_column].astype(str),
        frame[pred_column].astype(str),
        rownames=["Actual"],
        colnames=["Predicted"],
        dropna=False,
    )


def _error_rows(frame: pd.DataFrame) -> pd.DataFrame:
    major_mismatch = frame["category_gold"].astype(str) != frame["category"].astype(str)
    sub_mismatch = frame["subcategory_gold"].astype(str) != frame["subcategory"].astype(str)
    priority_mismatch = frame["priority_gold"].astype(str) != frame["priority"].astype(str)
    review_mismatch = _as_bool(frame["human_review_gold"]) != _as_bool(
        frame["requires_human_review"]
    )
    mask = major_mismatch | sub_mismatch | priority_mismatch | review_mismatch
    errors = frame.loc[mask].copy()
    if errors.empty:
        return errors

    def classify_error(row: pd.Series) -> str:
        if str(row["category_gold"]) != str(row["category"]):
            return "major_category"
        if str(row["subcategory_gold"]) != str(row["subcategory"]):
            return "subcategory_boundary"
        if str(row["priority_gold"]) != str(row["priority"]):
            return "priority"
        return "human_review"

    errors["suggested_error_type"] = errors.apply(classify_error, axis=1)
    errors["error_type"] = ""
    preferred = [
        "ticket_id",
        "customer_message",
        "category_gold",
        "subcategory_gold",
        "category",
        "subcategory",
        "priority_gold",
        "priority",
        "human_review_gold",
        "requires_human_review",
        "reason",
        "suggested_error_type",
        "error_type",
    ]
    return errors[[column for column in preferred if column in errors.columns]]


def _usage_summary(frame: pd.DataFrame) -> dict[str, Any]:
    def total(column: str) -> int:
        if column not in frame.columns:
            return 0
        return int(pd.to_numeric(frame[column], errors="coerce").fillna(0).sum())

    models = []
    if "model" in frame.columns:
        models = sorted(
            value
            for value in frame["model"].dropna().astype(str).unique().tolist()
            if value
        )
    total_cost: float | None = None
    if "estimated_cost_usd" in frame.columns:
        cost = pd.to_numeric(frame["estimated_cost_usd"], errors="coerce")
        if cost.notna().any():
            total_cost = float(cost.sum())

    return {
        "input_tokens": total("input_tokens"),
        "cached_input_tokens": total("cached_input_tokens"),
        "output_tokens": total("output_tokens"),
        "estimated_cost_usd": total_cost,
        "models": models,
    }


def _version_summary(frame: pd.DataFrame) -> dict[str, str]:
    mapping = {
        "dataset": "label_version",
        "prompt": "prompt_version",
        "taxonomy": "taxonomy_version",
    }
    result: dict[str, str] = {}
    for key, column in mapping.items():
        if column not in frame.columns:
            result[key] = "unknown"
            continue
        values = sorted(frame[column].dropna().astype(str).unique().tolist())
        result[key] = ", ".join(values) if values else "unknown"
    return result


def evaluate_results(
    results: pd.DataFrame,
    *,
    include_provisional: bool = False,
) -> EvaluationReport:
    """Evaluate analyzed rows. Provisional labels are excluded by default."""
    _require_columns(results, GOLD_COLUMNS)
    _require_columns(results, PREDICTION_COLUMNS)

    eligible = results.copy()
    if not include_provisional:
        eligible = eligible[
            eligible["label_status"].astype(str).str.lower().eq(PUBLISHABLE_LABEL_STATUS)
        ]

    successful = eligible[
        eligible["analysis_status"].astype(str).str.lower().eq("success")
    ].copy()

    if successful.empty:
        reason = (
            "검수 완료(reviewed) 라벨이 없습니다."
            if not include_provisional
            else "성공적으로 분석된 평가 행이 없습니다."
        )
        raise EvaluationDataError(reason)

    major, _ = _classification_metrics(
        successful["category_gold"], successful["category"]
    )
    subcategory, per_subcategory = _classification_metrics(
        successful["subcategory_gold"], successful["subcategory"]
    )
    high_risk = _high_risk_metrics(successful)
    human_review = _binary_metrics(
        successful["human_review_gold"], successful["requires_human_review"]
    )

    return EvaluationReport(
        rows_total=len(results),
        rows_evaluated=len(successful),
        rows_failed=int(
            eligible["analysis_status"].astype(str).str.lower().eq("failed").sum()
        ),
        publishable=not include_provisional,
        major=major,
        subcategory=subcategory,
        high_risk=high_risk,
        human_review=human_review,
        usage=_usage_summary(successful),
        versions=_version_summary(successful),
        major_confusion=_confusion(successful, "category_gold", "category"),
        subcategory_confusion=_confusion(
            successful, "subcategory_gold", "subcategory"
        ),
        errors=_error_rows(successful),
        per_subcategory=per_subcategory,
    )


def summary_dict(report: EvaluationReport) -> dict[str, Any]:
    """Return JSON-serializable summary metadata without large matrix payloads."""
    return {
        "rows_total": report.rows_total,
        "rows_evaluated": report.rows_evaluated,
        "rows_failed": report.rows_failed,
        "publishable": report.publishable,
        "major": report.major,
        "subcategory": report.subcategory,
        "high_risk": report.high_risk,
        "human_review": report.human_review,
        "usage": report.usage,
        "versions": report.versions,
    }
