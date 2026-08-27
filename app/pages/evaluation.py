from pathlib import Path

import pandas as pd
import streamlit as st

st.title("Evaluation")
st.caption("Measure whether the VOC classifier is reliable enough for CX operations use.")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Major Category Accuracy", "Not measured")
col2.metric("Subcategory Accuracy", "Not measured")
col3.metric("High Priority Recall", "Not measured")
col4.metric("Human Review Recall", "Not measured")

st.info(
    "v0.1에서는 평가 체계부터 먼저 정의합니다. 실제 수치는 사람이 라벨링한 gold dataset과 AI 예측을 비교한 뒤에만 표시합니다."
)

st.subheader("Planned Metrics")
metrics = pd.DataFrame(
    [
        {
            "metric": "Major Category Accuracy",
            "purpose": "배송/환불/상품정보 등 대분류가 정확한지 확인",
        },
        {
            "metric": "Subcategory Accuracy",
            "purpose": "배송완료 미수령/환불 지연 등 세부 원인이 정확한지 확인",
        },
        {
            "metric": "High Priority Recall",
            "purpose": "위험도가 높은 VOC를 놓치지 않는지 확인",
        },
        {
            "metric": "Human Review Recall",
            "purpose": "상담원 확인이 필요한 건을 놓치지 않는지 확인",
        },
    ]
)
st.dataframe(metrics, use_container_width=True, hide_index=True)

st.subheader("Evaluation Workflow")
st.markdown(
    "1. 사람이 VOC에 정답 라벨을 부여합니다.\n"
    "2. 동일한 VOC를 classifier에 입력합니다.\n"
    "3. Human label과 AI prediction을 비교합니다.\n"
    "4. 오분류 패턴을 taxonomy 또는 prompt 개선으로 연결합니다.\n"
    "5. 개선 전후 결과를 같은 gold dataset으로 재평가합니다."
)

eval_path = Path(__file__).resolve().parents[2] / "evals" / "evaluation_dataset.csv"
if eval_path.exists():
    eval_df = pd.read_csv(eval_path)
    st.subheader("Current Gold Dataset")
    st.caption(f"현재 seed label: {len(eval_df):,}건")
    st.dataframe(eval_df.head(20), use_container_width=True, hide_index=True)
