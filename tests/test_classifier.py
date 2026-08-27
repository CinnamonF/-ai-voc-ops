import pytest

from app.services import classifier
from app.services.llm import StructuredResponse, UsageMetadata


REQUIRED_FIELDS = {
    "category",
    "subcategory",
    "priority",
    "sentiment",
    "requires_human_review",
    "reason",
}


def structured(data):
    return StructuredResponse(
        data=data,
        usage=UsageMetadata(
            input_tokens=100,
            cached_input_tokens=20,
            output_tokens=30,
            model="gpt-5.6-luna",
            estimated_cost_usd=None,
        ),
    )


def test_classifier_returns_required_fields(monkeypatch):
    fake_result = {
        "category": "배송",
        "subcategory": "배송완료 미수령",
        "priority": "high",
        "sentiment": "negative",
        "requires_human_review": True,
        "reason": "배송완료 상태지만 고객이 실제 상품을 수령하지 못했습니다.",
    }

    monkeypatch.setattr(
        classifier,
        "create_structured_response",
        lambda **kwargs: structured(fake_result),
    )

    result = classifier.classify_voc(
        "배송 완료라고 나오는데 상품을 못 받았어요."
    )

    assert set(result) == REQUIRED_FIELDS
    assert result == fake_result


def test_empty_message_is_sent_to_human_review():
    result = classifier.classify_voc("   ")

    assert result["category"] == "클레임/기타"
    assert result["subcategory"] == "기타"
    assert result["requires_human_review"] is True


def test_invalid_category_subcategory_pair_is_rejected(monkeypatch):
    bad_result = {
        "category": "배송",
        "subcategory": "환불 지연",
        "priority": "normal",
        "sentiment": "neutral",
        "requires_human_review": False,
        "reason": "잘못된 조합을 테스트합니다.",
    }

    monkeypatch.setattr(
        classifier,
        "create_structured_response",
        lambda **kwargs: structured(bad_result),
    )

    with pytest.raises(classifier.VOCClassificationError):
        classifier.classify_voc("테스트 문의")


def test_high_risk_result_is_forced_to_human_review(monkeypatch):
    fake_result = {
        "category": "주문/결제",
        "subcategory": "중복 결제",
        "priority": "normal",
        "sentiment": "negative",
        "requires_human_review": False,
        "reason": "동일 주문이 두 번 결제됐습니다.",
    }
    monkeypatch.setattr(
        classifier,
        "create_structured_response",
        lambda **kwargs: structured(fake_result),
    )

    result = classifier.classify_voc("같은 주문이 두 번 결제됐어요")

    assert result["requires_human_review"] is True


def test_negative_sentiment_alone_does_not_force_human_review(monkeypatch):
    fake_result = {
        "category": "상품정보",
        "subcategory": "사용법",
        "priority": "normal",
        "sentiment": "negative",
        "requires_human_review": False,
        "reason": "사용 방법을 묻는 일반 문의입니다.",
    }
    monkeypatch.setattr(
        classifier,
        "create_structured_response",
        lambda **kwargs: structured(fake_result),
    )

    result = classifier.classify_voc("설명이 너무 어려운데 어떻게 써요?")

    assert result["requires_human_review"] is False


def test_missing_output_field_is_rejected(monkeypatch):
    incomplete = {
        "category": "배송",
        "subcategory": "배송 조회",
        "priority": "normal",
        "sentiment": "neutral",
        "requires_human_review": False,
    }
    monkeypatch.setattr(
        classifier,
        "create_structured_response",
        lambda **kwargs: structured(incomplete),
    )

    with pytest.raises(classifier.VOCClassificationError, match="누락"):
        classifier.classify_voc("배송 조회")
