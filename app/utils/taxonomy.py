"""Canonical VOC taxonomy for v0.1."""

TAXONOMY = {
    "배송": [
        {"subcategory": "출고 지연", "definition": "판매자 출고가 예정 시점보다 늦어진 문의", "example": "아직 발송이 안 됐어요"},
        {"subcategory": "배송 지연", "definition": "출고 후 택배 이동 또는 도착이 지연된 문의", "example": "택배가 며칠째 움직이지 않아요"},
        {"subcategory": "배송완료 미수령", "definition": "배송완료 상태지만 고객이 상품을 받지 못한 문의", "example": "배송 완료인데 못 받았어요"},
        {"subcategory": "배송 조회", "definition": "송장 또는 현재 배송 상태 확인 문의", "example": "송장 조회가 안 돼요"},
        {"subcategory": "배송지 변경", "definition": "주문 후 배송 주소 변경 요청", "example": "배송지 바꿀 수 있나요"},
        {"subcategory": "배송 중 분실", "definition": "배송 과정에서 상품 분실이 의심되거나 확인된 문의", "example": "택배가 중간에서 사라진 것 같아요"},
    ],
    "주문/결제": [
        {"subcategory": "결제 실패", "definition": "카드·간편결제 등 결제 승인 실패 문의", "example": "카드 결제가 안 됩니다"},
        {"subcategory": "중복 결제", "definition": "동일 주문 또는 금액이 두 번 이상 결제된 문의", "example": "두 번 결제됐어요"},
        {"subcategory": "주문 변경", "definition": "옵션·수량 등 주문 내용 변경 요청", "example": "옵션 바꾸고 싶어요"},
        {"subcategory": "주문 확인", "definition": "주문 접수 또는 주문 상태 확인 문의", "example": "주문이 정상 접수됐나요"},
        {"subcategory": "쿠폰/프로모션", "definition": "쿠폰·프로모션 조건 또는 적용 실패 문의", "example": "쿠폰 적용이 안 돼요"},
        {"subcategory": "가격/할인", "definition": "판매가·할인가·가격 차이에 대한 문의", "example": "왜 가격이 달라요"},
    ],
    "취소/환불": [
        {"subcategory": "주문 취소", "definition": "주문 전체 또는 일부 취소 요청", "example": "주문 취소하고 싶어요"},
        {"subcategory": "환불 지연", "definition": "취소·반품 이후 환불 처리가 지연된 문의", "example": "환불이 아직 안 됐어요"},
        {"subcategory": "부분 환불", "definition": "전체 결제금액 중 일부만 환불된 문의", "example": "일부만 환불됐어요"},
        {"subcategory": "환불 금액", "definition": "환불 금액 산정 또는 차액에 관한 문의", "example": "왜 이 금액만 환불됐나요"},
        {"subcategory": "취소 불가", "definition": "취소 제한 또는 취소 실패 이유에 관한 문의", "example": "왜 취소가 안 되나요"},
    ],
    "교환/반품": [
        {"subcategory": "단순 변심 반품", "definition": "상품 문제 없이 고객 변심으로 반품을 원하는 문의", "example": "마음이 바뀌어서 반품하고 싶어요"},
        {"subcategory": "상품 불량", "definition": "상품 기능·품질상 하자에 대한 문의", "example": "제품이 작동하지 않아요"},
        {"subcategory": "파손", "definition": "배송 또는 수령 과정에서 상품이 깨지거나 훼손된 문의", "example": "깨져서 왔어요"},
        {"subcategory": "오배송 상품", "definition": "주문과 다른 상품 또는 옵션을 받은 문의", "example": "다른 상품이 왔어요"},
        {"subcategory": "교환 절차", "definition": "교환 신청 방법·절차·진행상태 문의", "example": "교환은 어떻게 하나요"},
        {"subcategory": "반품비", "definition": "반품 배송비 또는 비용 부담 주체에 관한 문의", "example": "반품 배송비가 왜 발생하나요"},
    ],
    "상품정보": [
        {"subcategory": "사용법", "definition": "상품 사용 방법·순서·주의사항 문의", "example": "어떻게 사용하나요"},
        {"subcategory": "성분/소재", "definition": "원료·성분·재질 등에 대한 문의", "example": "어떤 성분이 들어있나요"},
        {"subcategory": "옵션/사이즈", "definition": "옵션·색상·용량·사이즈 선택 문의", "example": "어떤 옵션을 골라야 하나요"},
        {"subcategory": "재고/재입고", "definition": "품절 여부 또는 재입고 일정 문의", "example": "언제 재입고되나요"},
        {"subcategory": "호환/적합성", "definition": "다른 제품과의 호환 또는 특정 조건에서의 적합성 문의", "example": "이 제품과 같이 써도 되나요"},
        {"subcategory": "유통기한/보관", "definition": "유통기한·사용기한·보관방법 문의", "example": "유통기한이 어떻게 되나요"},
    ],
    "계정/서비스": [
        {"subcategory": "로그인", "definition": "로그인 실패 또는 인증 관련 문의", "example": "로그인이 안 돼요"},
        {"subcategory": "회원정보", "definition": "전화번호·주소·이메일 등 회원정보 변경 문의", "example": "전화번호 변경하고 싶어요"},
        {"subcategory": "포인트/적립금", "definition": "적립·사용·소멸된 포인트 관련 문의", "example": "적립금이 사라졌어요"},
        {"subcategory": "알림/메시지", "definition": "앱·문자·이메일 알림 수신 관련 문의", "example": "알림이 안 와요"},
    ],
    "클레임/기타": [
        {"subcategory": "반복 불만", "definition": "동일 문제로 여러 차례 문의했으나 해결되지 않은 불만", "example": "계속 문의했는데 해결이 안 됐어요"},
        {"subcategory": "정책 이의", "definition": "회사 정책 또는 처리 기준에 대한 이의제기", "example": "왜 이런 정책인가요"},
        {"subcategory": "상담 불만", "definition": "상담 응대 품질·태도·처리에 대한 불만", "example": "상담 대응이 불친절했어요"},
        {"subcategory": "개인정보/보안", "definition": "개인정보 노출·오사용·계정 보안 관련 문의", "example": "제 개인정보가 다른 사람에게 보였어요"},
        {"subcategory": "기타", "definition": "현재 taxonomy로 의미 있게 분류하기 어려운 문의", "example": "분류 기준에 없는 문의"},
    ],
}

PRIORITY_RULES = {
    "low": "정보성 문의로 즉시 처리 필요성이 낮은 경우",
    "normal": "일반적인 CS 처리 범위의 문의",
    "high": "금전·미수령·중복결제·반복 불만 등 빠른 확인이 필요한 경우",
    "critical": "개인정보·보안 또는 즉각적인 리스크 대응이 필요한 경우",
}

HUMAN_REVIEW_RULES = [
    "priority가 high 또는 critical인 경우",
    "개인정보/보안 관련 문의",
    "중복 결제·배송완료 미수령·반복 불만처럼 판단 또는 후속 조치가 필요한 경우",
    "AI 분류 확신도가 낮거나 기타로 분류되는 경우",
]
