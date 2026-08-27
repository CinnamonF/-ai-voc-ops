"""VOC classification service backed by OpenAI Structured Outputs."""

from __future__ import annotations

from typing import Any

try:
    from services.llm import create_structured_response
    from services.taxonomy import ALL_SUBCATEGORIES, TAXONOMY
except ModuleNotFoundError:
    from app.services.llm import create_structured_response
    from app.services.taxonomy import ALL_SUBCATEGORIES, TAXONOMY


OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "category": {"type": "string", "enum": list(TAXONOMY)},
        "subcategory": {"type": "string", "enum": ALL_SUBCATEGORIES},
        "priority": {
            "type": "string",
            "enum": ["low", "normal", "high", "critical"],
        },
        "sentiment": {
            "type": "string",
            "enum": ["positive", "neutral", "negative"],
        },
        "requires_human_review": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": [
        "category",
        "subcategory",
        "priority",
        "sentiment",
        "requires_human_review",
        "reason",
    ],
    "additionalProperties": False,
}

TAXONOMY_PROMPT = """
너는 이커머스 CS/CX 운영을 위한 VOC 분류기다.
고객 문의의 핵심 해결 대상 하나를 기준으로 아래 taxonomy 중 정확히 하나를 선택한다.

배송: 출고 지연 | 배송 지연 | 배송완료 미수령 | 배송 조회 | 배송지 변경 | 배송 중 분실
주문/결제: 결제 실패 | 중복 결제 | 주문 변경 | 주문 확인 | 쿠폰/프로모션 | 가격/할인
취소/환불: 주문 취소 | 환불 지연 | 부분 환불 | 환불 금액 | 취소 불가
교환/반품: 단순 변심 반품 | 상품 불량 | 파손 | 오배송 상품 | 교환 절차 | 반품비
상품정보: 사용법 | 성분/소재 | 옵션/사이즈 | 재고/재입고 | 호환/적합성 | 유통기한/보관
계정/서비스: 로그인 | 회원정보 | 포인트/적립금 | 알림/메시지
클레임/기타: 반복 불만 | 정책 이의 | 상담 불만 | 개인정보/보안 | 기타

경계 규칙:
- 시스템상 배송완료지만 고객이 받지 못했으면 '배송완료 미수령'이다.
- 배송 이동 과정에서 분실이 의심되면 '배송 중 분실'이다.
- 주문과 다른 상품/옵션이 도착했으면 '오배송 상품'이다.
- 기능·품질 하자는 '상품 불량', 깨짐·누수·찢김 같은 물리 훼손은 '파손'이다.
- 환불 시점 문제는 '환불 지연', 금액 산정 문제는 '환불 금액', 일부만 환불된 상태는 '부분 환불'이다.
- '기타'는 기존 분류로 의미 있게 설명할 수 없을 때만 사용한다.

보조 태그:
- low: 단순 정보성, normal: 일반 CS, high: 금전·미수령·중복결제·반복불만 등 빠른 확인 필요
- critical: 개인정보·보안 등 즉각적인 리스크 대응 필요
- high/critical, 개인정보/보안, 중복결제, 배송완료 미수령, 반복불만, 낮은 확신의 기타는 human review를 true로 둔다.
- sentiment는 고객 감정만 판단하며 category 결정 근거로 쓰지 않는다.
- reason은 한국어 한 문장으로 짧고 구체적으로 작성한다.
""".strip()


class VOCClassificationError(ValueError):
    """Raised when model output conflicts with the project taxonomy."""


def _validate_result(result: dict[str, Any]) -> dict[str, Any]:
    category = result.get("category")
    subcategory = result.get("subcategory")

    if category not in TAXONOMY:
        raise VOCClassificationError(f"알 수 없는 대분류: {category}")

    if subcategory not in TAXONOMY[category]:
        raise VOCClassificationError(
            "대분류/소분류 조합이 taxonomy와 일치하지 않습니다: "
            f"{category} > {subcategory}"
        )

    return result


def classify_voc(text: str) -> dict[str, Any]:
    """Classify one customer message into the fixed VOC taxonomy."""
    clean_text = str(text).strip()

    if not clean_text:
        return {
            "category": "클레임/기타",
            "subcategory": "기타",
            "priority": "normal",
            "sentiment": "neutral",
            "requires_human_review": True,
            "reason": "문의 내용이 비어 있어 사람이 확인해야 합니다.",
        }

    result = create_structured_response(
        instructions=TAXONOMY_PROMPT,
        user_input=f"고객 문의:\n{clean_text}",
        schema_name="voc_classification",
        schema=OUTPUT_SCHEMA,
    )
    return _validate_result(result)
