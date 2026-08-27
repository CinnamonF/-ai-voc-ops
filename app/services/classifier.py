"""VOC classification service backed by OpenAI Structured Outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.services.llm import UsageMetadata, create_structured_response
from app.services.taxonomy import (
    ALL_SUBCATEGORIES,
    HUMAN_REVIEW_SUBCATEGORIES,
    PRIORITIES,
    SENTIMENTS,
    TAXONOMY,
)


OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "category": {"type": "string", "enum": list(TAXONOMY)},
        "subcategory": {"type": "string", "enum": ALL_SUBCATEGORIES},
        "priority": {
            "type": "string",
            "enum": list(PRIORITIES),
        },
        "sentiment": {
            "type": "string",
            "enum": list(SENTIMENTS),
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

OUTPUT_FIELDS = frozenset(OUTPUT_SCHEMA["required"])
TAXONOMY_LABELS = "\n".join(
    f"{category}: {' | '.join(subcategories)}"
    for category, subcategories in TAXONOMY.items()
)

TAXONOMY_PROMPT = f"""
너는 이커머스 CS/CX 운영을 위한 VOC 분류기다.
고객 문의의 핵심 해결 대상 하나를 기준으로 아래 taxonomy 중 정확히 하나를 선택한다.

{TAXONOMY_LABELS}

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


@dataclass(frozen=True)
class VOCAnalysis:
    classification: dict[str, Any]
    usage: UsageMetadata


def _validate_result(result: dict[str, Any]) -> dict[str, Any]:
    fields = set(result)
    missing = OUTPUT_FIELDS - fields
    unexpected = fields - OUTPUT_FIELDS
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"누락: {', '.join(sorted(missing))}")
        if unexpected:
            details.append(f"허용되지 않은 필드: {', '.join(sorted(unexpected))}")
        raise VOCClassificationError("분류 결과 필드가 올바르지 않습니다. " + "; ".join(details))

    category = result.get("category")
    subcategory = result.get("subcategory")
    if category not in TAXONOMY:
        raise VOCClassificationError(f"알 수 없는 대분류: {category}")
    if subcategory not in TAXONOMY[category]:
        raise VOCClassificationError(
            "대분류/소분류 조합이 taxonomy와 일치하지 않습니다: "
            f"{category} > {subcategory}"
        )

    priority = result.get("priority")
    sentiment = result.get("sentiment")
    review = result.get("requires_human_review")
    reason = result.get("reason")
    if priority not in PRIORITIES:
        raise VOCClassificationError(f"알 수 없는 우선순위: {priority}")
    if sentiment not in SENTIMENTS:
        raise VOCClassificationError(f"알 수 없는 감정 라벨: {sentiment}")
    if not isinstance(review, bool):
        raise VOCClassificationError("requires_human_review는 boolean이어야 합니다.")
    if not isinstance(reason, str) or not reason.strip():
        raise VOCClassificationError("reason은 비어 있지 않은 문자열이어야 합니다.")

    validated = dict(result)
    validated["reason"] = reason.strip()
    if priority in {"high", "critical"} or subcategory in HUMAN_REVIEW_SUBCATEGORIES:
        validated["requires_human_review"] = True
    return validated


def _empty_classification() -> dict[str, Any]:
    return {
        "category": "클레임/기타",
        "subcategory": "기타",
        "priority": "normal",
        "sentiment": "neutral",
        "requires_human_review": True,
        "reason": "문의 내용이 비어 있어 사람이 확인해야 합니다.",
    }


def classify_voc_with_usage(text: str) -> VOCAnalysis:
    """Classify one message and retain measured API usage separately."""
    clean_text = str(text).strip()
    if not clean_text:
        return VOCAnalysis(
            classification=_empty_classification(),
            usage=UsageMetadata.no_api_call(),
        )

    response = create_structured_response(
        instructions=TAXONOMY_PROMPT,
        user_input=clean_text,
        schema_name="voc_classification",
        schema=OUTPUT_SCHEMA,
    )
    return VOCAnalysis(
        classification=_validate_result(response.data),
        usage=response.usage,
    )


def classify_voc(text: str) -> dict[str, Any]:
    """Classify one customer message into the fixed VOC taxonomy."""
    return classify_voc_with_usage(text).classification
