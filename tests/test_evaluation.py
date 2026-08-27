import pandas as pd
import pytest

from app.services.evaluation import EvaluationDataError, evaluate_results


def _fixture() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ticket_id": "1", "customer_message": "배송 완료인데 못 받았어요",
                "category_gold": "배송", "subcategory_gold": "배송완료 미수령",
                "priority_gold": "high", "sentiment_gold": "negative", "human_review_gold": True,
                "label_status": "reviewed", "label_version": "test-v1",
                "category": "배송", "subcategory": "배송완료 미수령", "priority": "high",
                "sentiment": "negative", "requires_human_review": True, "analysis_status": "success",
                "reason": "배송완료 상태지만 미수령", "prompt_version": "v0.1",
                "taxonomy_version": "v0.1", "input_tokens": 100, "cached_input_tokens": 20,
                "output_tokens": 10, "model": "test-model", "estimated_cost_usd": 0.001,
            },
            {
                "ticket_id": "2", "customer_message": "일부만 환불됐어요",
                "category_gold": "취소/환불", "subcategory_gold": "부분 환불",
                "priority_gold": "high", "sentiment_gold": "negative", "human_review_gold": True,
                "label_status": "reviewed", "label_version": "test-v1",
                "category": "취소/환불", "subcategory": "환불 금액", "priority": "normal",
                "sentiment": "negative", "requires_human_review": False, "analysis_status": "success",
                "reason": "금액 차이 문의로 판단", "prompt_version": "v0.1",
                "taxonomy_version": "v0.1", "input_tokens": 110, "cached_input_tokens": 20,
                "output_tokens": 12, "model": "test-model", "estimated_cost_usd": 0.001,
            },
        ]
    )


def test_evaluation_metrics_and_errors():
    report = evaluate_results(_fixture())
    assert report.rows_evaluated == 2
    assert report.major["accuracy"] == 1.0
    assert report.subcategory["accuracy"] == 0.5
    assert report.high_risk["recall"] == 0.5
    assert report.human_review["recall"] == 0.5
    assert len(report.errors) == 1
    assert report.usage["input_tokens"] == 210


def test_provisional_labels_are_not_publishable_by_default():
    frame = _fixture()
    frame["label_status"] = "provisional"
    with pytest.raises(EvaluationDataError, match="reviewed"):
        evaluate_results(frame)


def test_provisional_labels_can_be_explicitly_explored():
    frame = _fixture()
    frame["label_status"] = "provisional"
    report = evaluate_results(frame, include_provisional=True)
    assert report.publishable is False
    assert report.rows_evaluated == 2


def test_failed_rows_are_excluded_from_quality_metrics():
    frame = _fixture()
    frame.loc[1, "analysis_status"] = "failed"
    report = evaluate_results(frame)
    assert report.rows_evaluated == 1
    assert report.rows_failed == 1
    assert report.subcategory["accuracy"] == 1.0
