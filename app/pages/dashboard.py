from pathlib import Path

import pandas as pd
import streamlit as st

from app.services.batch import summarize_results

st.title("AI VOC Ops")
st.caption("AI-powered VOC Intelligence & CX Operations System")

results = st.session_state.get("voc_results")

if not isinstance(results, pd.DataFrame) or results.empty:
    st.info("아직 분석된 VOC가 없습니다. VOC Analyzer에서 CSV를 업로드하고 분석을 실행하세요.")

    sample_path = Path(__file__).resolve().parents[2] / "data" / "samples" / "sample_voc.csv"
    if sample_path.exists():
        st.subheader("Sample input")
        sample = pd.read_csv(sample_path)
        st.dataframe(sample, width="stretch", hide_index=True)
    st.stop()

required_columns = {
    "category",
    "subcategory",
    "priority",
    "sentiment",
    "requires_human_review",
}
missing = required_columns - set(results.columns)
if missing:
    st.error(f"분석 결과에 필요한 컬럼이 없습니다: {', '.join(sorted(missing))}")
    st.stop()

summary = summarize_results(results)
if "analysis_status" in results.columns:
    analyzed_results = results[results["analysis_status"].eq("success")]
elif "analysis_error" in results.columns:
    analyzed_results = results[results["analysis_error"].isna()]
else:
    analyzed_results = results

col1, col2, col3, col4 = st.columns(4)
col1.metric("Input VOC", f"{summary.input_rows:,}")
col2.metric("Analyzed", f"{summary.successful_rows:,}")
col3.metric("Failed", f"{summary.failed_rows:,}")
col4.metric("Human Review", f"{summary.human_review_rows:,}")

if analyzed_results.empty:
    st.warning("성공한 분석 결과가 없어 분류 차트를 표시할 수 없습니다.")
    st.stop()

left, right = st.columns([3, 2])

with left:
    st.subheader("VOC Category Distribution")
    category_counts = (
        analyzed_results["category"]
        .fillna("미분류")
        .value_counts()
        .rename_axis("category")
        .to_frame("count")
    )
    st.bar_chart(category_counts)

with right:
    st.subheader("Top Issues")
    top_issues = (
        analyzed_results.assign(
            category=analyzed_results["category"].fillna("미분류"),
            subcategory=analyzed_results["subcategory"].fillna("미분류"),
        )
        .groupby(["category", "subcategory"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(10)
    )
    st.dataframe(top_issues, width="stretch", hide_index=True)

st.subheader("Recent analysis result")
st.dataframe(analyzed_results.head(50), width="stretch", hide_index=True)
