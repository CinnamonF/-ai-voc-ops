import pandas as pd
import streamlit as st

from utils.taxonomy import HUMAN_REVIEW_RULES, PRIORITY_RULES, TAXONOMY

st.title("VOC Taxonomy")
st.caption("The classification system used by AI VOC Ops v0.1.")

st.info(
    "Taxonomy는 모델이 임의의 라벨을 만드는 것을 막고, 동일한 운영 기준으로 VOC를 집계·평가하기 위한 기준입니다."
)

for category, items in TAXONOMY.items():
    with st.expander(f"{category} · {len(items)} subcategories"):
        st.dataframe(pd.DataFrame(items), use_container_width=True, hide_index=True)

st.subheader("Priority Rules")
priority_df = pd.DataFrame(
    [{"priority": key, "definition": value} for key, value in PRIORITY_RULES.items()]
)
st.dataframe(priority_df, use_container_width=True, hide_index=True)

st.subheader("Human Review Rules")
for rule in HUMAN_REVIEW_RULES:
    st.markdown(f"- {rule}")
