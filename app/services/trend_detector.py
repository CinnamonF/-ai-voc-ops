"""Deterministic VOC aggregation, comparison, and spike detection.

The service is intentionally offline and explainable. It consumes the columns
already produced by :mod:`app.services.batch` (plus a configurable timestamp)
and returns tables that a later Trend UI can render without reimplementing the
business rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from app.services.trend_detector_alerts import ALERT_COLUMNS, empty_alerts, make_alerts

SUPPORTED_FREQUENCIES = ("daily", "weekly")
SUPPORTED_DIMENSIONS = ("category", "subcategory")

# These are operational guardrails, not statistically tuned thresholds.
DEFAULT_MIN_CURRENT_COUNT = 5
DEFAULT_MIN_ABSOLUTE_INCREASE = 3
DEFAULT_BASELINE_PERIODS = 4

PERIOD_METRIC_COLUMNS = (
    "issue_key",
    "category",
    "subcategory",
    "period_start",
    "period_id",
    "count",
    "evidence_ticket_ids",
    "evidence_row_indices",
)

@dataclass(frozen=True)
class TrendAnalysisResult:
    """Output of :func:`analyze_trends`.

    ``period_metrics`` contains one row for every issue/period combination,
    including zero-count periods between the first and last usable period.
    ``alerts`` contains the latest-period comparison for every observed issue;
    rows are ranked deterministically with alerts first.
    """

    period_metrics: pd.DataFrame
    alerts: pd.DataFrame
    data_quality: dict[str, Any]

    @property
    def comparisons(self) -> pd.DataFrame:
        """Alias for callers that prefer the comparison terminology."""

        return self.alerts


def _empty_period_metrics() -> pd.DataFrame:
    return pd.DataFrame(columns=PERIOD_METRIC_COLUMNS)


def _missing_mask(series: pd.Series) -> pd.Series:
    """Return a robust missing/blank mask for text-like input columns."""

    return series.isna() | series.astype("string").str.strip().eq("").fillna(False)


def _period_start(parsed: pd.Series, frequency: str) -> pd.Series:
    normalized = parsed.dt.normalize()
    if frequency == "daily":
        return normalized
    # Monday is the explicit start of a weekly period. Subtracting the
    # weekday avoids pandas Period timezone loss and keeps UTC timestamps.
    return normalized - pd.to_timedelta(normalized.dt.weekday, unit="D")


def _period_id(value: pd.Timestamp) -> str:
    return value.strftime("%Y-%m-%d")


def _issue_key(category: str, subcategory: str | None, dimension: str) -> str:
    if dimension == "category":
        return category
    return f"{category} > {subcategory}"


def _validate_options(
    frame: pd.DataFrame,
    timestamp_column: str,
    frequency: str,
    dimension: str,
    min_current_count: int,
    min_absolute_increase: int,
    baseline_periods: int,
) -> tuple[str, str]:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    if timestamp_column not in frame.columns:
        raise ValueError(f"timestamp column not found: {timestamp_column}")

    normalized_frequency = str(frequency).strip().lower()
    if normalized_frequency not in SUPPORTED_FREQUENCIES:
        allowed = ", ".join(SUPPORTED_FREQUENCIES)
        raise ValueError(f"frequency must be one of: {allowed}")

    normalized_dimension = str(dimension).strip().lower()
    if normalized_dimension not in SUPPORTED_DIMENSIONS:
        allowed = ", ".join(SUPPORTED_DIMENSIONS)
        raise ValueError(f"dimension must be one of: {allowed}")

    required = {"category"}
    if normalized_dimension == "subcategory":
        required.add("subcategory")
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("trend dimension columns not found: " + ", ".join(missing))

    for name, value in (
        ("min_current_count", min_current_count),
        ("min_absolute_increase", min_absolute_increase),
        ("baseline_periods", baseline_periods),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")

    return normalized_frequency, normalized_dimension


def _parse_timestamps(series: pd.Series) -> tuple[pd.Series, int, int]:
    missing = _missing_mask(series)
    try:
        parsed = pd.to_datetime(series, errors="coerce", utc=True)
    except (TypeError, ValueError):
        parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns, UTC]")
    parsed = pd.Series(parsed, index=series.index)
    invalid = parsed.isna() & ~missing
    return parsed, int(invalid.sum()), int(missing.sum())


def _stable_evidence(rows: pd.DataFrame) -> tuple[list[str], list[int]]:
    row_indices = [int(value) for value in rows["_trend_row_index"].tolist()]
    if "ticket_id" not in rows.columns:
        return [], row_indices

    ticket_ids = [
        str(value)
        for value in rows["ticket_id"].tolist()
        if not pd.isna(value) and str(value).strip()
    ]
    return ticket_ids, row_indices


def _make_period_metrics(
    usable: pd.DataFrame,
    periods: pd.DatetimeIndex,
    *,
    dimension: str,
) -> pd.DataFrame:
    issue_columns = ["category"] if dimension == "category" else ["category", "subcategory"]
    issue_rows = (
        usable.loc[:, issue_columns]
        .drop_duplicates()
        .sort_values(issue_columns, kind="stable")
        .itertuples(index=False, name=None)
    )
    output: list[dict[str, Any]] = []
    for issue_values in issue_rows:
        issue_category = str(issue_values[0])
        issue_subcategory = (
            str(issue_values[1]) if dimension == "subcategory" else None
        )
        mask = usable["category"].eq(issue_category)
        if dimension == "subcategory":
            mask &= usable["subcategory"].eq(issue_subcategory)
        issue_frame = usable.loc[mask]

        for period in periods:
            period_rows = issue_frame.loc[
                issue_frame["_trend_period_start"].eq(period)
            ]
            ticket_ids, row_indices = _stable_evidence(period_rows)
            output.append(
                {
                    "issue_key": _issue_key(
                        issue_category, issue_subcategory, dimension
                    ),
                    "category": issue_category,
                    "subcategory": issue_subcategory,
                    "period_start": period,
                    "period_id": _period_id(period),
                    "count": int(len(period_rows)),
                    "evidence_ticket_ids": ticket_ids,
                    "evidence_row_indices": row_indices,
                }
            )

    if not output:
        return _empty_period_metrics()
    return pd.DataFrame(output, columns=PERIOD_METRIC_COLUMNS)


def analyze_trends(
    frame: pd.DataFrame,
    timestamp_column: str,
    *,
    frequency: str = "weekly",
    dimension: str = "subcategory",
    min_current_count: int = DEFAULT_MIN_CURRENT_COUNT,
    min_absolute_increase: int = DEFAULT_MIN_ABSOLUTE_INCREASE,
    baseline_periods: int = DEFAULT_BASELINE_PERIODS,
) -> TrendAnalysisResult:
    """Analyze VOC volume changes by period and issue dimension.

    The latest available period is compared with the immediately preceding
    calendar period. Weekly periods run Monday through Sunday in UTC. Rows
    with invalid/missing timestamps or missing selected dimensions are omitted
    from aggregation and represented in ``data_quality`` instead.
    """

    normalized_frequency, normalized_dimension = _validate_options(
        frame,
        timestamp_column,
        frequency,
        dimension,
        min_current_count,
        min_absolute_increase,
        baseline_periods,
    )

    work = frame.copy(deep=True).reset_index(drop=True)
    work["_trend_row_index"] = range(len(work))
    parsed, invalid_timestamps, missing_timestamps = _parse_timestamps(
        work[timestamp_column]
    )
    work["_trend_period_start"] = _period_start(parsed, normalized_frequency)

    category_missing = _missing_mask(work["category"])
    subcategory_missing = (
        _missing_mask(work["subcategory"])
        if "subcategory" in work.columns
        else pd.Series(False, index=work.index)
    )
    dimension_missing = (
        category_missing
        if normalized_dimension == "category"
        else category_missing | subcategory_missing
    )
    usable_mask = parsed.notna() & ~dimension_missing
    usable = work.loc[usable_mask].copy()
    usable["category"] = usable["category"].astype("string").str.strip()
    if "subcategory" in usable.columns:
        usable["subcategory"] = usable["subcategory"].astype("string").str.strip()

    quality: dict[str, Any] = {
        "input_rows": int(len(frame)),
        "usable_rows": int(len(usable)),
        "invalid_timestamp_rows": invalid_timestamps,
        "missing_timestamp_rows": missing_timestamps,
        "missing_category_rows": int(category_missing.sum()),
        "missing_subcategory_rows": int(subcategory_missing.sum()),
        "periods_available": 0,
        "observed_periods": 0,
        "frequency": normalized_frequency,
        "dimension": normalized_dimension,
        "baseline_periods_required": baseline_periods,
        "history_state": "insufficient_history",
    }
    if usable.empty:
        return TrendAnalysisResult(_empty_period_metrics(), empty_alerts(), quality)

    min_period = usable["_trend_period_start"].min()
    max_period = usable["_trend_period_start"].max()
    periods = pd.date_range(
        min_period,
        max_period,
        freq="D" if normalized_frequency == "daily" else "7D",
        tz="UTC",
    )
    quality["periods_available"] = int(len(periods))
    quality["observed_periods"] = int(usable["_trend_period_start"].nunique())
    quality["history_state"] = (
        "sufficient" if len(periods) >= baseline_periods + 1 else "insufficient_history"
    )

    period_metrics = _make_period_metrics(
        usable, periods, dimension=normalized_dimension
    )
    alerts = make_alerts(
        period_metrics,
        periods,
        baseline_periods=baseline_periods,
        min_current_count=min_current_count,
        min_absolute_increase=min_absolute_increase,
        dimension=normalized_dimension,
    )
    return TrendAnalysisResult(period_metrics, alerts, quality)


__all__ = [
    "ALERT_COLUMNS",
    "DEFAULT_BASELINE_PERIODS",
    "DEFAULT_MIN_ABSOLUTE_INCREASE",
    "DEFAULT_MIN_CURRENT_COUNT",
    "PERIOD_METRIC_COLUMNS",
    "SUPPORTED_DIMENSIONS",
    "SUPPORTED_FREQUENCIES",
    "TrendAnalysisResult",
    "analyze_trends",
]
