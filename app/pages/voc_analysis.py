import io

import pandas as pd
import streamlit as st

from app.services.batch import analyze_batch, summarize_results
from app.services.llm import get_model_name, is_api_configured


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

model_name = get_model_name()
if is_api_configured():
    st.success(f"OpenAI API connected · model: `{model_name}`")
else:
    st.warning(
        "OPENAI_API_KEY가 설정되어 있지 않습니다. "
        "CSV 미리보기는 가능하지만 AI 분석 실행 전 API 키 설정이 필요합니다."
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
    st.dataframe(raw_df.head(20), width="stretch", hide_index=True)

    message_column = st.selectbox(
        "Customer message column",
        options=list(raw_df.columns),
        index=list(raw_df.columns).index("customer_message")
        if "customer_message" in raw_df.columns
        else 0,
    )

    if st.button("Analyze VOC", type="primary", width="stretch"):
        if not is_api_configured():
            st.error(
                "OPENAI_API_KEY가 없어 분석을 시작할 수 없습니다. "
                ".env.example을 참고해 키를 설정하세요."
            )
            st.stop()

        progress = st.progress(0, text="VOC를 구조화하고 있습니다...")

        def update_progress(index: int, total: int) -> None:
            progress.progress(
                index / total,
                text=f"VOC 분석 중 · {index:,}/{total:,}",
            )

        result_df = analyze_batch(
            raw_df,
            message_column,
            on_progress=update_progress,
        )
        progress.empty()
        st.session_state["voc_results"] = result_df

        summary = summarize_results(result_df)
        if summary.failed_rows:
            st.warning(
                f"{summary.input_rows:,}건 중 {summary.failed_rows:,}건은 분석에 실패했습니다. "
                "`analysis_error` 컬럼을 확인하세요."
            )
        else:
            st.success(f"{summary.successful_rows:,}건의 VOC 분석을 완료했습니다.")

results = st.session_state.get("voc_results")
if not isinstance(results, pd.DataFrame) or results.empty:
    st.stop()

st.divider()
st.subheader("Analysis Result")

summary = summarize_results(results)
metric_columns = st.columns(5)
metric_columns[0].metric("Input rows", f"{summary.input_rows:,}")
metric_columns[1].metric("Analyzed", f"{summary.successful_rows:,}")
metric_columns[2].metric("Failed", f"{summary.failed_rows:,}")
metric_columns[3].metric("Human review", f"{summary.human_review_rows:,}")
metric_columns[4].metric("High / critical", f"{summary.high_priority_rows:,}")

filter_col1, filter_col2, filter_col3 = st.columns(3)
with filter_col1:
    category_options = sorted(
        results["category"].dropna().astype(str).unique().tolist()
    )
    selected_categories = st.multiselect("Category", category_options)
with filter_col2:
    priority_options = sorted(
        results["priority"].dropna().astype(str).unique().tolist()
    )
    selected_priorities = st.multiselect("Priority", priority_options)
with filter_col3:
    review_filter = st.selectbox("Human Review", ["All", "Yes", "No"])

filtered = results.copy()
if selected_categories:
    filtered = filtered[
        filtered["category"].astype(str).isin(selected_categories)
    ]
if selected_priorities:
    filtered = filtered[
        filtered["priority"].astype(str).isin(selected_priorities)
    ]
if review_filter != "All":
    review_bool = (
        filtered["requires_human_review"]
        .astype(str)
        .str.lower()
        .isin(["true", "1", "yes"])
    )
    filtered = filtered[review_bool if review_filter == "Yes" else ~review_bool]

st.caption(f"Showing {len(filtered):,} of {len(results):,} rows")
st.dataframe(filtered, width="stretch", hide_index=True)

csv_bytes = filtered.to_csv(index=False).encode("utf-8-sig")
st.download_button(
    "Download analyzed CSV",
    data=csv_bytes,
    file_name="voc_analysis_result.csv",
    mime="text/csv",
    width="stretch",
)
