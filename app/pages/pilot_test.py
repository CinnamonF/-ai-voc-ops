from __future__ import annotations

from uuid import uuid4

import pandas as pd
import streamlit as st

from app.services.classifier import classify_voc_with_usage
from app.services.llm import get_model_name, is_api_configured
from app.services.pilot import (
    FeedbackPersistenceError,
    PilotConfigurationError,
    build_feedback_record,
    get_single_analysis_limit,
    get_text_char_limit,
    is_feedback_store_configured,
    redact_pii,
    save_feedback,
)
from app.services.taxonomy import PRIORITIES, SENTIMENTS, TAXONOMY

st.title("Pilot Test")
st.caption("Try one VOC, inspect the AI decision, and leave structured feedback for model evaluation.")

st.warning(
    "테스트용 또는 비식별화된 문의만 입력하세요. 실제 고객 이름, 전화번호, 이메일, 주문번호 등 "
    "개인정보가 포함된 원문은 사용하지 마세요."
)

if "pilot_session_id" not in st.session_state:
    st.session_state["pilot_session_id"] = str(uuid4())
if "pilot_feedback_rows" not in st.session_state:
    st.session_state["pilot_feedback_rows"] = []
if "pilot_analysis_count" not in st.session_state:
    st.session_state["pilot_analysis_count"] = 0

try:
    session_limit = get_single_analysis_limit()
    text_limit = get_text_char_limit()
except PilotConfigurationError as exc:
    st.error(str(exc))
    st.stop()

backend = "Supabase persistent storage" if is_feedback_store_configured() else "Session-only storage"
top1, top2, top3 = st.columns(3)
top1.metric("Model", get_model_name())
top2.metric("Feedback backend", backend)
top3.metric(
    "Session analyses",
    f"{st.session_state['pilot_analysis_count']}/{session_limit}",
)

if not is_api_configured():
    st.error("OPENAI_API_KEY가 설정되어 있지 않아 Pilot Test를 실행할 수 없습니다.")
    st.stop()

message = st.text_area(
    "Customer VOC",
    placeholder="예: 배송 완료라고 나오는데 실제로 상품을 받지 못했습니다",
    height=140,
    max_chars=text_limit,
)

confirmed_safe_input = st.checkbox(
    "이 문의는 테스트용 또는 비식별화된 내용이며 실제 고객 개인정보를 포함하지 않습니다."
)

analysis_disabled = (
    not message.strip()
    or not confirmed_safe_input
    or st.session_state["pilot_analysis_count"] >= session_limit
)

if st.button("Analyze VOC", type="primary", width="stretch", disabled=analysis_disabled):
    try:
        with st.spinner("VOC를 분석하고 있습니다..."):
            safe_message = redact_pii(message)
            analysis = classify_voc_with_usage(safe_message)
    except Exception as exc:
        st.error(f"분석에 실패했습니다: {exc}")
    else:
        st.session_state["pilot_analysis_count"] += 1
        st.session_state["pilot_current"] = {
            "message": safe_message,
            "analysis": analysis,
            "analysis_id": str(uuid4()),
        }

if st.session_state["pilot_analysis_count"] >= session_limit:
    st.info("이 세션의 Pilot Test 분석 한도에 도달했습니다.")

current = st.session_state.get("pilot_current")
if not current:
    st.stop()

analysis = current["analysis"]
classification = analysis.classification
usage = analysis.usage

st.divider()
st.subheader("AI Prediction")

result_cols = st.columns(4)
result_cols[0].metric("Category", classification["category"])
result_cols[1].metric("Subcategory", classification["subcategory"])
result_cols[2].metric("Priority", classification["priority"])
result_cols[3].metric(
    "Human review",
    "Yes" if classification["requires_human_review"] else "No",
)

st.write(f"**Sentiment:** {classification['sentiment']}")
st.write(f"**Reason:** {classification['reason']}")
st.caption(
    "이메일·전화번호·긴 숫자 패턴은 API 호출 전에 마스킹됩니다. "
    f"model={usage.model or get_model_name()} · input={usage.input_tokens:,} · "
    f"cached={usage.cached_input_tokens:,} · output={usage.output_tokens:,}"
)

st.divider()
st.subheader("Tester Feedback")
st.caption(
    "맞다고 판단한 결과도 저장하면 positive validation 데이터가 됩니다. "
    "저장되는 문의 텍스트에는 이메일·전화번호·긴 숫자 패턴을 추가로 마스킹합니다."
)

correctness = st.radio(
    "이 분류가 맞나요?",
    ["맞음", "틀림"],
    horizontal=True,
    key=f"correctness_{current['analysis_id']}",
)
is_correct = correctness == "맞음"

category_options = list(TAXONOMY)
pred_category = classification["category"]
category_index = category_options.index(pred_category)

if is_correct:
    corrected_category = pred_category
    corrected_subcategory = classification["subcategory"]
else:
    corrected_category = st.selectbox(
        "Correct category",
        category_options,
        index=category_index,
        key=f"category_{current['analysis_id']}",
    )
    subcategory_options = list(TAXONOMY[corrected_category])
    predicted_subcategory = classification["subcategory"]
    subcategory_index = (
        subcategory_options.index(predicted_subcategory)
        if predicted_subcategory in subcategory_options
        else 0
    )
    corrected_subcategory = st.selectbox(
        "Correct subcategory",
        subcategory_options,
        index=subcategory_index,
        key=f"subcategory_{current['analysis_id']}",
    )

priority_options = list(PRIORITIES)
corrected_priority = st.selectbox(
    "Correct priority",
    priority_options,
    index=priority_options.index(classification["priority"]),
    key=f"priority_{current['analysis_id']}",
)

sentiment_options = list(SENTIMENTS)
corrected_sentiment = st.selectbox(
    "Correct sentiment",
    sentiment_options,
    index=sentiment_options.index(classification["sentiment"]),
    key=f"sentiment_{current['analysis_id']}",
)

corrected_human_review = st.checkbox(
    "Human review required",
    value=bool(classification["requires_human_review"]),
    key=f"review_{current['analysis_id']}",
)

feedback_note = st.text_area(
    "Optional note",
    placeholder="왜 이 라벨이 맞거나 틀렸다고 판단했는지 간단히 적어주세요.",
    max_chars=1000,
    key=f"note_{current['analysis_id']}",
)

feedback_consent = st.checkbox(
    "이 테스트 결과를 AI VOC Ops 평가·개선용 피드백으로 저장하는 데 동의합니다.",
    key=f"consent_{current['analysis_id']}",
)

if st.button(
    "Submit feedback",
    width="stretch",
    disabled=not feedback_consent,
    key=f"submit_{current['analysis_id']}",
):
    try:
        record = build_feedback_record(
            session_id=st.session_state["pilot_session_id"],
            message=current["message"],
            analysis=analysis,
            is_correct=is_correct,
            corrected_category=corrected_category,
            corrected_subcategory=corrected_subcategory,
            corrected_priority=corrected_priority,
            corrected_sentiment=corrected_sentiment,
            corrected_human_review=corrected_human_review,
            feedback_note=feedback_note,
        )
    except Exception as exc:
        st.error(f"피드백을 만들지 못했습니다: {exc}")
    else:
        row = record.as_dict()
        st.session_state["pilot_feedback_rows"].append(row)
        if is_feedback_store_configured():
            try:
                save_feedback(record)
            except FeedbackPersistenceError as exc:
                st.warning(
                    f"세션에는 저장했지만 영구 저장에는 실패했습니다: {exc} "
                    "아래 CSV를 다운로드해 보관할 수 있습니다."
                )
            else:
                st.success("피드백을 저장했습니다. 감사합니다.")
        else:
            st.success(
                "이 세션에 피드백을 저장했습니다. 영구 저장소가 연결되지 않았으므로 "
                "세션 종료 전 CSV를 다운로드하세요."
            )

feedback_rows = st.session_state["pilot_feedback_rows"]
if feedback_rows:
    st.divider()
    st.subheader("This session's feedback")
    feedback_df = pd.DataFrame(feedback_rows)
    st.dataframe(
        feedback_df[
            [
                "created_at",
                "message_redacted",
                "prediction_category",
                "prediction_subcategory",
                "is_correct",
                "corrected_category",
                "corrected_subcategory",
            ]
        ],
        width="stretch",
        hide_index=True,
    )
    st.download_button(
        "Download session feedback CSV",
        data=feedback_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="pilot_feedback_session.csv",
        mime="text/csv",
        width="stretch",
    )
