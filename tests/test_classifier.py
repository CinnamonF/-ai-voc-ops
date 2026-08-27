from app.services import classifier


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
        lambda **kwargs: fake_result,
    )

    result = classifier.classify_voc(
        "배송 완료라고 나오는데 상품을 못 받았어요."
    )

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
        lambda **kwargs: bad_result,
    )

    try:
        classifier.classify_voc("테스트 문의")
    except classifier.VOCClassificationError:
        pass
    else:
        raise AssertionError("Invalid taxonomy pair must be rejected")
