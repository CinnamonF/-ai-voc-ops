"""Private comparison and ranking helpers for the trend detector service."""

from __future__ import annotations

from math import isfinite
from typing import Any

import pandas as pd

ALERT_COLUMNS = (
    "rank",
    "issue_key",
    "category",
    "subcategory",
    "current_period_start",
    "current_period_id",
    "previous_period_start",
    "previous_period_id",
    "current_count",
    "previous_count",
    "absolute_change",
    "percentage_change",
    "comparison_state",
    "baseline_state",
    "baseline_periods",
    "baseline_mean",
    "baseline_std",
    "baseline_z_score",
    "volume_component",
    "absolute_change_component",
    "relative_change_component",
    "baseline_deviation_component",
    "spike_score",
    "min_current_count",
    "min_absolute_increase",
    "guardrail_passed",
    "is_alert",
    "alert_class",
    "alert_reason",
    "evidence_ticket_ids",
    "evidence_row_indices",
)


def empty_alerts() -> pd.DataFrame:
    return pd.DataFrame(columns=ALERT_COLUMNS)


def _baseline_values(
    issue_metrics: pd.DataFrame,
    current_period: pd.Timestamp,
    baseline_periods: int,
) -> tuple[str, list[int]]:
    history = issue_metrics.loc[
        issue_metrics["period_start"].lt(current_period), "count"
    ].tolist()
    history = [int(value) for value in history[-baseline_periods:]]
    if len(history) < baseline_periods:
        return "insufficient_history", history
    if len(set(history)) == 1:
        return "zero_variance", history
    return "sufficient", history


def _score_components(
    *,
    current_count: int,
    absolute_change: int,
    percentage_change: float | None,
    baseline_z_score: float | None,
    baseline_state: str,
    min_current_count: int,
) -> tuple[float, float, float, float, float | None]:
    volume = round(current_count / min_current_count, 6)
    absolute = float(max(absolute_change, 0))
    if percentage_change is None:
        relative = 1.0 if current_count > 0 and absolute_change > 0 else 0.0
    else:
        relative = max(percentage_change, 0.0) / 100.0
    relative = round(relative, 6)
    baseline = (
        round(max(baseline_z_score, 0.0), 6)
        if baseline_z_score is not None
        else 0.0
    )
    # Insufficient history means there is no defensible anomaly score. The
    # direct comparison and guardrail fields remain available for operators.
    if baseline_state == "insufficient_history":
        return volume, absolute, relative, baseline, None
    score = round(volume + absolute + relative + baseline, 6)
    return volume, absolute, relative, baseline, score


def _alert_row(
    issue_metrics: pd.DataFrame,
    current_period: pd.Timestamp,
    previous_period: pd.Timestamp | None,
    *,
    baseline_periods: int,
    min_current_count: int,
    min_absolute_increase: int,
) -> dict[str, Any]:
    current_row = issue_metrics.loc[
        issue_metrics["period_start"].eq(current_period)
    ].iloc[0]
    current_count = int(current_row["count"])
    if previous_period is None:
        previous_count = 0
    else:
        previous_count = int(
            issue_metrics.loc[
                issue_metrics["period_start"].eq(previous_period), "count"
            ].iloc[0]
        )
    absolute_change = current_count - previous_count

    if previous_period is None:
        comparison_state = "no_previous_period"
        percentage_change = None
    elif previous_count == 0 and current_count > 0:
        comparison_state = "new_issue"
        percentage_change = None
    elif previous_count == 0:
        comparison_state = "no_change"
        percentage_change = None
    else:
        comparison_state = "comparison"
        percentage_change = round(absolute_change / previous_count * 100, 6)

    baseline_state, history = _baseline_values(
        issue_metrics, current_period, baseline_periods
    )
    baseline_mean = float(sum(history) / len(history)) if history else None
    baseline_std: float | None = None
    baseline_z_score: float | None = None
    if baseline_state == "sufficient":
        assert baseline_mean is not None
        variance = sum((value - baseline_mean) ** 2 for value in history) / len(history)
        baseline_std = variance**0.5
        if baseline_std > 0:
            baseline_z_score = (current_count - baseline_mean) / baseline_std
            if not isfinite(baseline_z_score):
                baseline_z_score = None
    elif baseline_state == "zero_variance":
        baseline_std = 0.0

    volume, absolute, relative, baseline, score = _score_components(
        current_count=current_count,
        absolute_change=absolute_change,
        percentage_change=percentage_change,
        baseline_z_score=baseline_z_score,
        baseline_state=baseline_state,
        min_current_count=min_current_count,
    )
    guardrail_passed = bool(
        current_count >= min_current_count
        and absolute_change >= min_absolute_increase
        and previous_period is not None
    )
    if guardrail_passed and comparison_state == "new_issue":
        alert_class = "emerging"
        alert_reason = "new emerging issue"
    elif guardrail_passed:
        alert_class = "spike"
        alert_reason = "volume spike"
    else:
        alert_class = "normal"
        if comparison_state == "no_previous_period":
            alert_reason = "no prior period for comparison"
        elif absolute_change > 0 and current_count < min_current_count:
            alert_reason = "below minimum-volume guardrail"
        elif absolute_change > 0 and absolute_change < min_absolute_increase:
            alert_reason = "below minimum-increase guardrail"
        else:
            alert_reason = "no meaningful increase"

    return {
        "issue_key": current_row["issue_key"],
        "category": current_row["category"],
        "subcategory": current_row["subcategory"],
        "current_period_start": current_period,
        "current_period_id": _period_id(current_period),
        "previous_period_start": previous_period,
        "previous_period_id": _period_id(previous_period) if previous_period is not None else None,
        "current_count": current_count,
        "previous_count": previous_count,
        "absolute_change": absolute_change,
        "percentage_change": percentage_change,
        "comparison_state": comparison_state,
        "baseline_state": baseline_state,
        "baseline_periods": len(history),
        "baseline_mean": baseline_mean,
        "baseline_std": baseline_std,
        "baseline_z_score": baseline_z_score,
        "volume_component": volume,
        "absolute_change_component": absolute,
        "relative_change_component": relative,
        "baseline_deviation_component": baseline,
        "spike_score": score,
        "min_current_count": min_current_count,
        "min_absolute_increase": min_absolute_increase,
        "guardrail_passed": guardrail_passed,
        "is_alert": guardrail_passed,
        "alert_class": alert_class,
        "alert_reason": alert_reason,
        "evidence_ticket_ids": list(current_row["evidence_ticket_ids"]),
        "evidence_row_indices": list(current_row["evidence_row_indices"]),
    }


def _period_id(value: pd.Timestamp) -> str:
    return value.strftime("%Y-%m-%d")


def make_alerts(
    period_metrics: pd.DataFrame,
    periods: pd.DatetimeIndex,
    *,
    baseline_periods: int,
    min_current_count: int,
    min_absolute_increase: int,
    dimension: str,
) -> pd.DataFrame:
    """Build latest-period comparisons and deterministic alert ranking."""

    if period_metrics.empty:
        return empty_alerts()
    current_period = periods[-1]
    previous_period = periods[-2] if len(periods) >= 2 else None
    issue_columns = ["category"] if dimension == "category" else ["category", "subcategory"]
    rows: list[dict[str, Any]] = []
    for _, issue_metrics in period_metrics.groupby(issue_columns, sort=False, dropna=False):
        rows.append(
            _alert_row(
                issue_metrics,
                current_period,
                previous_period,
                baseline_periods=baseline_periods,
                min_current_count=min_current_count,
                min_absolute_increase=min_absolute_increase,
            )
        )
    alerts = pd.DataFrame(rows)
    alerts = alerts.sort_values(
        by=["is_alert", "spike_score", "current_count", "absolute_change", "issue_key"],
        ascending=[False, False, False, False, True],
        na_position="last",
        kind="stable",
    ).reset_index(drop=True)
    alerts.insert(0, "rank", range(1, len(alerts) + 1))
    alerts["percentage_change"] = alerts["percentage_change"].astype(object)
    alerts.loc[
        alerts["comparison_state"].isin(["new_issue", "no_change", "no_previous_period"]),
        "percentage_change",
    ] = None
    return alerts.loc[:, ALERT_COLUMNS]


__all__ = ["ALERT_COLUMNS", "empty_alerts", "make_alerts"]
