import pandas as pd
import streamlit as st

from app.services.taxonomy import HUMAN_REVIEW_RULES, PRIORITY_RULES, TAXONOMY_DETAILS
from app.utils.config import VERSION

st.title("VOC Taxonomy")
st.caption(f"The classification system used by AI VOC Ops v{VERSION}.")

st.info(
    "Taxonomy는 모델이 임의의 라벨을 만드는 것을 막고, 동일한 운영 기준으로 VOC를 집계·평가하기 위한 기준입니다."
)

for category, items in TAXONOMY_DETAILS.items():
    with st.expander(f"{category} · {len(items)} subcategories"):
        st.dataframe(pd.DataFrame(items), width="stretch", hide_index=True)

st.subheader("Priority Rules")
priority_df = pd.DataFrame(
    [{"priority": key, "definition": value} for key, value in PRIORITY_RULES.items()]
)
st.dataframe(priority_df, width="stretch", hide_index=True)

st.subheader("Human Review Rules")
for rule in HUMAN_REVIEW_RULES:
    st.markdown(f"- {rule}")
