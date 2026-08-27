import pandas as pd

from app.services.batch import analyze_batch, summarize_results
from app.services.classifier import VOCAnalysis
from app.services.llm import UsageMetadata


def successful_analysis() -> VOCAnalysis:
    return VOCAnalysis(
        classification={
            "category": "배송",
            "subcategory": "배송완료 미수령",
            "priority": "high",
            "sentiment": "negative",
            "requires_human_review": True,
            "reason": "배송완료 상태지만 상품을 받지 못했습니다.",
        },
        usage=UsageMetadata(
            input_tokens=100,
            cached_input_tokens=20,
            output_tokens=25,
            model="gpt-5.6-luna",
            estimated_cost_usd=0.0001,
        ),
    )


def test_batch_preserves_rows_and_isolates_one_failure():
    source = pd.DataFrame(
        {
            "ticket_id": ["VOC-1", "VOC-2"],
            "message": ["배송 완료인데 못 받았어요", "API 실패를 유도"],
        }
    )

    def classifier(text: str) -> VOCAnalysis:
        if "실패" in text:
            raise RuntimeError("sensitive provider detail")
        return successful_analysis()

    result = analyze_batch(source, "message", classifier=classifier)

    assert result["ticket_id"].tolist() == ["VOC-1", "VOC-2"]
    assert result["analysis_status"].tolist() == ["success", "failed"]
    assert result.loc[0, "input_tokens"] == 100
    assert pd.isna(result.loc[1, "category"])
    assert result.loc[1, "analysis_error"] == (
        "예상하지 못한 분석 오류가 발생했습니다. 다시 시도하세요."
    )

    summary = summarize_results(result)
    assert summary.input_rows == 2
    assert summary.successful_rows == 1
    assert summary.failed_rows == 1
    assert summary.human_review_rows == 1
    assert summary.high_priority_rows == 1


def test_batch_replaces_stale_analysis_columns():
    source = pd.DataFrame(
        {
            "message": ["배송 완료인데 못 받았어요"],
            "category": ["stale"],
            "analysis_error": ["old error"],
        }
    )

    result = analyze_batch(
        source,
        "message",
        classifier=lambda text: successful_analysis(),
    )

    assert result.loc[0, "category"] == "배송"
    assert result.loc[0, "analysis_error"] is None


def test_batch_reports_missing_message_column():
    source = pd.DataFrame({"ticket_id": ["VOC-1"]})

    try:
        analyze_batch(source, "message")
    except ValueError as exc:
        assert "메시지 컬럼" in str(exc)
    else:
        raise AssertionError("Missing message column must be rejected")
