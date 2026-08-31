from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from app.services.trend_detector import analyze_trends


def _rows(
    counts: list[int],
    *,
    subcategory: str = "배송완료 미수령",
    start: date = date(2026, 7, 6),
    prefix: str = "T",
    include_ticket_id: bool = True,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for period_index, count in enumerate(counts):
        period_start = start + timedelta(days=7 * period_index)
        for offset in range(count):
            row: dict[str, object] = {
                "created_at": period_start.isoformat(),
                "category": "배송",
                "subcategory": subcategory,
                "customer_message": f"sample {prefix}-{period_index}-{offset}",
            }
            if include_ticket_id:
                row["ticket_id"] = f"{prefix}-{period_index}-{offset}"
            rows.append(row)
    return pd.DataFrame(rows)


def _alert(result, issue_key: str = "배송 > 배송완료 미수령") -> pd.Series:
    return result.alerts.loc[result.alerts["issue_key"].eq(issue_key)].iloc[0]


def test_daily_and_weekly_aggregation_use_explicit_period_starts():
    frame = pd.DataFrame(
        {
            "created_at": [
                "2026-08-03T00:30:00Z",
                "2026-08-04T00:30:00Z",
                "2026-08-09T23:59:00Z",
                "2026-08-10T00:01:00Z",
            ],
            "category": ["배송"] * 4,
            "subcategory": ["배송완료 미수령"] * 4,
            "ticket_id": ["D-1", "D-2", "D-3", "D-4"],
        }
    )

    daily = analyze_trends(frame, "created_at", frequency="daily", dimension="category")
    daily_counts = daily.period_metrics.set_index("period_id")["count"].to_dict()
    assert {key: daily_counts[key] for key in ("2026-08-03", "2026-08-04", "2026-08-09", "2026-08-10")} == {
        "2026-08-03": 1,
        "2026-08-04": 1,
        "2026-08-09": 1,
        "2026-08-10": 1,
    }
    assert daily_counts["2026-08-05"] == 0
    assert daily.data_quality["periods_available"] == 8

    weekly = analyze_trends(frame, "created_at", frequency="weekly", dimension="category")
    weekly_counts = weekly.period_metrics.set_index("period_id")["count"].to_dict()
    assert weekly_counts == {"2026-08-03": 3, "2026-08-10": 1}
    assert weekly.period_metrics["period_start"].dt.weekday.eq(0).all()


def test_subcategory_dimension_preserves_category_and_issue_key():
    frame = pd.concat(
        [
            _rows([2, 3], subcategory="배송완료 미수령", prefix="A"),
            _rows([1, 4], subcategory="배송 지연", prefix="B"),
        ],
        ignore_index=True,
    )
    result = analyze_trends(frame, "created_at", dimension="subcategory")
    assert set(result.period_metrics["category"]) == {"배송"}
    assert set(result.alerts["issue_key"]) == {
        "배송 > 배송완료 미수령",
        "배송 > 배송 지연",
    }


def test_comparison_reports_change_and_finite_percentage():
    result = analyze_trends(_rows([5, 8]), "created_at", baseline_periods=1)
    alert = _alert(result)
    assert alert["current_count"] == 8
    assert alert["previous_count"] == 5
    assert alert["absolute_change"] == 3
    assert alert["percentage_change"] == 60.0
    assert alert["comparison_state"] == "comparison"
    assert bool(alert["is_alert"])


def test_zero_previous_count_is_explicit_new_issue_without_infinity():
    base = _rows([1, 1, 1, 1, 1], subcategory="배송 지연", prefix="BASE")
    emerging = _rows([0, 0, 0, 0, 10], subcategory="배송완료 미수령", prefix="NEW")
    result = analyze_trends(
        pd.concat([base, emerging], ignore_index=True),
        "created_at",
    )
    alert = _alert(result)
    assert alert["comparison_state"] == "new_issue"
    assert alert["previous_count"] == 0
    assert alert["current_count"] == 10
    assert pd.isna(alert["percentage_change"])
    assert alert["alert_class"] == "emerging"
    assert bool(alert["is_alert"])


def test_two_zero_periods_have_no_meaningful_percentage():
    base = _rows([1, 1, 1], subcategory="배송 지연", prefix="BASE")
    zeros = _rows([1, 0, 0], subcategory="배송완료 미수령", prefix="ZERO")
    result = analyze_trends(pd.concat([base, zeros], ignore_index=True), "created_at")
    alert = _alert(result)
    assert alert["comparison_state"] == "no_change"
    assert pd.isna(alert["percentage_change"])
    assert not bool(alert["is_alert"])


def test_minimum_volume_protects_tiny_denominator():
    result = analyze_trends(_rows([1, 2]), "created_at")
    alert = _alert(result)
    assert alert["percentage_change"] == 100.0
    assert alert["current_count"] < alert["min_current_count"]
    assert not bool(alert["is_alert"])
    assert alert["alert_reason"] == "below minimum-volume guardrail"


def test_clear_spike_is_alert_and_stable_series_is_not():
    spike = analyze_trends(_rows([5, 6, 5, 6, 25]), "created_at")
    stable = analyze_trends(_rows([5, 6, 5, 6, 5]), "created_at")
    assert _alert(spike)["alert_class"] == "spike"
    assert bool(_alert(spike)["is_alert"])
    assert not bool(_alert(stable)["is_alert"])
    assert _alert(stable)["alert_class"] == "normal"


def test_multiple_issue_ranking_prefers_strong_spike_then_moderate_then_stable():
    frames = [
        _rows([5, 6, 5, 6, 25], prefix="STRONG", subcategory="배송완료 미수령"),
        _rows([5, 5, 5, 6, 10], prefix="MODERATE", subcategory="배송 지연"),
        _rows([5, 5, 5, 5, 5], prefix="STABLE", subcategory="배송 조회"),
    ]
    result = analyze_trends(pd.concat(frames, ignore_index=True), "created_at")
    assert result.alerts["issue_key"].tolist() == [
        "배송 > 배송완료 미수령",
        "배송 > 배송 지연",
        "배송 > 배송 조회",
    ]
    assert result.alerts["is_alert"].tolist() == [True, True, False]
    assert result.alerts["spike_score"].iloc[0] > result.alerts["spike_score"].iloc[1]


def test_zero_variance_baseline_is_safe_and_finite():
    result = analyze_trends(_rows([5, 5, 5, 5, 10]), "created_at")
    alert = _alert(result)
    assert alert["baseline_state"] == "zero_variance"
    assert alert["baseline_std"] == 0.0
    assert alert["baseline_z_score"] is None
    assert isinstance(alert["spike_score"], float)


def test_insufficient_history_is_explicit_without_fabricated_score():
    result = analyze_trends(_rows([5, 8]), "created_at")
    alert = _alert(result)
    assert result.data_quality["history_state"] == "insufficient_history"
    assert alert["baseline_state"] == "insufficient_history"
    assert alert["spike_score"] is None
    assert bool(alert["is_alert"])


def test_bad_and_missing_timestamps_are_isolated_from_valid_rows():
    frame = _rows([5, 8])
    frame.loc[len(frame)] = {
        "created_at": "not-a-date",
        "category": "배송",
        "subcategory": "배송완료 미수령",
        "ticket_id": "BAD-1",
    }
    frame.loc[len(frame)] = {
        "created_at": "",
        "category": "배송",
        "subcategory": "배송완료 미수령",
        "ticket_id": "MISSING-1",
    }
    result = analyze_trends(frame, "created_at")
    assert result.data_quality["invalid_timestamp_rows"] == 1
    assert result.data_quality["missing_timestamp_rows"] == 1
    assert result.data_quality["usable_rows"] == len(frame) - 2
    assert _alert(result)["current_count"] == 8


def test_missing_dimensions_are_counted_and_excluded():
    frame = _rows([5, 8])
    bad_index = len(frame)
    frame.loc[bad_index, ["category", "subcategory"]] = None
    result = analyze_trends(frame, "created_at")
    assert result.data_quality["missing_category_rows"] == 1
    assert result.data_quality["missing_subcategory_rows"] == 1
    assert result.data_quality["usable_rows"] == len(frame) - 1


def test_alerts_retain_current_period_evidence_ticket_ids():
    result = analyze_trends(_rows([5, 8]), "created_at")
    alert = _alert(result)
    assert alert["evidence_ticket_ids"] == [f"T-1-{index}" for index in range(8)]
    current_metrics = result.period_metrics.loc[
        result.period_metrics["period_id"].eq("2026-07-13")
    ].iloc[0]
    assert current_metrics["evidence_ticket_ids"] == [f"T-1-{index}" for index in range(8)]


def test_missing_ticket_id_uses_stable_source_row_indices():
    result = analyze_trends(_rows([2, 2], include_ticket_id=False), "created_at")
    alert = _alert(result)
    assert alert["evidence_ticket_ids"] == []
    assert alert["evidence_row_indices"] == [2, 3]


def test_input_frame_is_not_mutated():
    frame = _rows([5, 8])
    before = frame.copy(deep=True)
    before_columns = frame.columns.tolist()
    analyze_trends(frame, "created_at")
    pd.testing.assert_frame_equal(frame, before)
    assert frame.columns.tolist() == before_columns


def test_invalid_options_are_rejected():
    frame = _rows([1])
    for kwargs in (
        {"frequency": "monthly"},
        {"dimension": "priority"},
        {"min_current_count": 0},
        {"baseline_periods": True},
    ):
        try:
            analyze_trends(frame, "created_at", **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid options should fail: {kwargs}")
