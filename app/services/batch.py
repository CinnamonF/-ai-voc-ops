"""Batch VOC analysis with row-level failure isolation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pandas as pd

from app.services.classifier import VOCAnalysis, classify_voc_with_usage
from app.services.llm import (
    LLMConfigurationError,
    LLMRequestError,
    LLMResponseError,
)

CLASSIFICATION_COLUMNS = (
    "category",
    "subcategory",
    "priority",
    "sentiment",
    "requires_human_review",
    "reason",
)
USAGE_COLUMNS = (
    "input_tokens",
    "cached_input_tokens",
    "output_tokens",
    "model",
    "estimated_cost_usd",
)
RESULT_COLUMNS = CLASSIFICATION_COLUMNS + USAGE_COLUMNS + (
    "analysis_status",
    "analysis_error",
)


@dataclass(frozen=True)
class BatchSummary:
    input_rows: int
    successful_rows: int
    failed_rows: int
    human_review_rows: int
    high_priority_rows: int


ProgressCallback = Callable[[int, int], None]
Classifier = Callable[[str], VOCAnalysis]


def _safe_error_message(exc: Exception) -> str:
    safe_errors = (
        LLMConfigurationError,
        LLMRequestError,
        LLMResponseError,
        ValueError,
    )
    if isinstance(exc, safe_errors):
        return str(exc)[:300]
    return "예상하지 못한 분석 오류가 발생했습니다. 다시 시도하세요."


def _failure_result(exc: Exception) -> dict[str, Any]:
    return {
        **{column: None for column in CLASSIFICATION_COLUMNS},
        **{column: None for column in USAGE_COLUMNS},
        "analysis_status": "failed",
        "analysis_error": _safe_error_message(exc),
    }


def analyze_batch(
    source: pd.DataFrame,
    message_column: str,
    *,
    classifier: Classifier = classify_voc_with_usage,
    on_progress: ProgressCallback | None = None,
) -> pd.DataFrame:
    """Analyze every row while retaining source fields and isolating failures."""
    if message_column not in source.columns:
        raise ValueError(f"메시지 컬럼을 찾을 수 없습니다: {message_column}")

    base = source.drop(
        columns=[column for column in RESULT_COLUMNS if column in source.columns],
        errors="ignore",
    ).reset_index(drop=True)
    messages = source[message_column].fillna("").astype(str).tolist()
    rows: list[dict[str, Any]] = []
    total = len(messages)

    for index, text in enumerate(messages, start=1):
        try:
            analysis = classifier(text)
            row = {
                **analysis.classification,
                **analysis.usage.as_columns(),
                "analysis_status": "success",
                "analysis_error": None,
            }
        except Exception as exc:
            row = _failure_result(exc)
        rows.append(row)
        if on_progress is not None:
            on_progress(index, total)

    return pd.concat([base, pd.DataFrame(rows)], axis=1)


def summarize_results(results: pd.DataFrame) -> BatchSummary:
    """Calculate operational counts without treating failed rows as predictions."""
    if results.empty:
        return BatchSummary(0, 0, 0, 0, 0)

    if "analysis_status" in results.columns:
        successful = results["analysis_status"].eq("success")
    elif "analysis_error" in results.columns:
        successful = results["analysis_error"].isna()
    else:
        successful = pd.Series(True, index=results.index)

    review = (
        results.get("requires_human_review", pd.Series(False, index=results.index))
        .astype(str)
        .str.lower()
        .isin(["true", "1", "yes"])
    )
    high_priority = (
        results.get("priority", pd.Series("", index=results.index))
        .astype(str)
        .str.lower()
        .isin(["high", "critical"])
    )
    successful_rows = int(successful.sum())
    return BatchSummary(
        input_rows=len(results),
        successful_rows=successful_rows,
        failed_rows=len(results) - successful_rows,
        human_review_rows=int((successful & review).sum()),
        high_priority_rows=int((successful & high_priority).sum()),
    )
