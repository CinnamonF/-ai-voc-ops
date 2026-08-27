import io

import pandas as pd
import streamlit as st

from services.classifier import classify_voc

ANALYSIS_COLUMNS = [
    "category",
    "subcategory",
    "priority",
    "sentiment",
    "requires_human_review",
    "reason",
]


def read_uploaded_csv(uploaded_file) -> pd.DataFrame:
    raw = uploaded_file.getvalue()
    for encoding in ("utf-8-sig", "utf-8", "cp949"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("지원되는 인코딩(UTF-8/CP949)으로 CSV를 읽을 수 없습니다.")


st.title("VOC Analyzer")
st.caption("Upload customer-support VOC data and convert it into structured CX operations fields.")

st.warning(
    "현재 v0.1 classifier는 scaffold 상태입니다. UI와 데이터 파이프라인을 먼저 연결했으며, "
    "실제 AI 분류 로직은 다음 단계에서 구현합니다."
)

uploaded_file = st.file_uploader("Upload VOC CSV", type=["csv"])

if uploaded_file is not None:
    try:
        raw_df = read_uploaded_csv(uploaded_file)
    except Exception as exc:
        st.error(f"CSV를 읽는 중 오류가 발생했습니다: {exc}")
        st.stop()

    if raw_df.empty:
        st.warning("CSV에 데이터가 없습니다.")
        st.stop()

    st.subheader("Input preview")
    st.dataframe(raw_df.head(20), use_container_width=True, hide_index=True)

    message_column = st.selectbox(
        "Customer message column",
        options=list(raw_df.columns),
        index=list(raw_df.columns).index("customer_message") if "customer_message" in raw_df.columns else 0,
    )

    if st.button("Analyze VOC", type="primary", use_container_width=True):
        with st.spinner("VOC를 구조화하고 있습니다..."):
            predictions = [
                classify_voc(text)
                for text in raw_df[message_column].fillna("").astype(str)
            ]
            base_df = raw_df.drop(
                columns=[column for column in ANALYSIS_COLUMNS if column in raw_df.columns],
                errors="ignore",
            ).reset_index(drop=True)
            prediction_df = pd.DataFrame(predictions)
            result_df = pd.concat([base_df, prediction_df], axis=1)
            st.session_state["voc_results"] = result_df
        st.success(f"{len(result_df):,}건의 VOC 분석 파이프라인을 실행했습니다.")

results = st.session_state.get("voc_results")
if not isinstance(results, pd.DataFrame) or results.empty:
    st.stop()

st.divider()
st.subheader("Analysis Result")

filter_col1, filter_col2, filter_col3 = st.columns(3)
with filter_col1:
    category_options = sorted(results["category"].dropna().astype(str).unique().tolist())
    selected_categories = st.multiselect("Category", category_options)
with filter_col2:
    priority_options = sorted(results["priority"].dropna().astype(str).unique().tolist())
    selected_priorities = st.multiselect("Priority", priority_options)
with filter_col3:
    review_filter = st.selectbox("Human Review", ["All", "Yes", "No"])

filtered = results.copy()
if selected_categories:
    filtered = filtered[filtered["category"].astype(str).isin(selected_categories)]
if selected_priorities:
    filtered = filtered[filtered["priority"].astype(str).isin(selected_priorities)]
if review_filter != "All":
    review_bool = filtered["requires_human_review"].astype(str).str.lower().isin(["true", "1", "yes"])
    filtered = filtered[review_bool if review_filter == "Yes" else ~review_bool]

st.caption(f"Showing {len(filtered):,} of {len(results):,} rows")
st.dataframe(filtered, use_container_width=True, hide_index=True)

csv_bytes = filtered.to_csv(index=False).encode("utf-8-sig")
st.download_button(
    "Download analyzed CSV",
    data=csv_bytes,
    file_name="voc_analysis_result.csv",
    mime="text/csv",
    use_container_width=True,
)
