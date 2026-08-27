from app.services.classifier import classify_voc


def test_classifier_returns_required_fields():
    result = classify_voc("배송 완료라고 나오는데 상품을 못 받았어요.")
    required = {
        "category",
        "subcategory",
        "priority",
        "sentiment",
        "requires_human_review",
    }
    assert required.issubset(result.keys())
