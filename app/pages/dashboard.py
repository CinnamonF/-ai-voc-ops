from pathlib import Path

import pandas as pd
import streamlit as st

st.title("AI VOC Ops")
st.caption("AI-powered VOC Intelligence & CX Operations System")

results = st.session_state.get("voc_results")

if not isinstance(results, pd.DataFrame) or results.empty:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total VOC", "—")
    col2.metric("High Priority", "—")
    col3.metric("Human Review", "—")
    col4.metric("Negative VOC", "—")

    st.info("아직 분석된 VOC가 없습니다. VOC Analyzer에서 CSV를 업로드하고 분석을 실행하세요.")

    sample_path = Path(__file__).resolve().parents[2] / "data" / "samples" / "sample_voc.csv"
    if sample_path.exists():
        st.subheader("Sample input")
        sample = pd.read_csv(sample_path)
        st.dataframe(sample, use_container_width=True, hide_index=True)
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

total = len(results)
high_priority = results["priority"].astype(str).str.lower().isin(["high", "critical"]).sum()
human_review = results["requires_human_review"].astype(str).str.lower().isin(["true", "1", "yes"]).sum()
negative = results["sentiment"].astype(str).str.lower().eq("negative").sum()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total VOC", f"{total:,}")
col2.metric("High Priority", f"{high_priority:,}")
col3.metric("Human Review", f"{human_review:,}")
col4.metric("Negative VOC", f"{negative:,}")

left, right = st.columns([3, 2])

with left:
    st.subheader("VOC Category Distribution")
    category_counts = (
        results["category"]
        .fillna("미분류")
        .value_counts()
        .rename_axis("category")
        .to_frame("count")
    )
    st.bar_chart(category_counts)

with right:
    st.subheader("Top Issues")
    top_issues = (
        results.assign(
            category=results["category"].fillna("미분류"),
            subcategory=results["subcategory"].fillna("미분류"),
        )
        .groupby(["category", "subcategory"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(10)
    )
    st.dataframe(top_issues, use_container_width=True, hide_index=True)

st.subheader("Recent analysis result")
st.dataframe(results.head(50), use_container_width=True, hide_index=True)
