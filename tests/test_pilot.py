from app.services.classifier import VOCAnalysis
from app.services.llm import UsageMetadata
from app.services.pilot import (
    build_feedback_record,
    get_batch_row_limit,
    redact_pii,
)


def _analysis() -> VOCAnalysis:
    return VOCAnalysis(
        classification={
            "category": "배송",
            "subcategory": "배송완료 미수령",
            "priority": "high",
            "sentiment": "negative",
            "requires_human_review": True,
            "reason": "배송완료 상태이지만 상품을 받지 못한 문의입니다.",
        },
        usage=UsageMetadata(
            input_tokens=100,
            cached_input_tokens=40,
            output_tokens=30,
            model="gpt-test",
            estimated_cost_usd=0.001,
        ),
    )


def test_redact_pii_masks_common_identifiers():
    text = "email me at user@example.com or 010-1234-5678 order 123456789"
    redacted = redact_pii(text)
    assert "user@example.com" not in redacted
    assert "010-1234-5678" not in redacted
    assert "123456789" not in redacted
    assert "[EMAIL]" in redacted
    assert "[PHONE]" in redacted
    assert "[NUMBER]" in redacted


def test_feedback_record_keeps_provenance_and_redacts():
    record = build_feedback_record(
        session_id="session-1",
        message="배송 완료인데 못 받았어요 010-1234-5678",
        analysis=_analysis(),
        is_correct=True,
        corrected_category="배송",
        corrected_subcategory="배송완료 미수령",
        corrected_priority="high",
        corrected_sentiment="negative",
        corrected_human_review=True,
        feedback_note="맞는 분류",
    )
    assert "[PHONE]" in record.message_redacted
    assert record.prediction_category == "배송"
    assert record.corrected_subcategory == "배송완료 미수령"
    assert record.model == "gpt-test"
    assert record.prompt_version
    assert record.taxonomy_version


def test_feedback_rejects_invalid_category_subcategory_pair():
    try:
        build_feedback_record(
            session_id="session-1",
            message="환불 문의",
            analysis=_analysis(),
            is_correct=False,
            corrected_category="배송",
            corrected_subcategory="환불 지연",
            corrected_priority="high",
            corrected_sentiment="negative",
            corrected_human_review=True,
        )
    except ValueError as exc:
        assert "taxonomy" in str(exc)
    else:
        raise AssertionError("invalid category/subcategory pair should fail")


def test_batch_limit_env(monkeypatch):
    monkeypatch.setenv("PILOT_MAX_BATCH_ROWS", "25")
    assert get_batch_row_limit() == 25
