import streamlit as st

st.set_page_config(
    page_title="AI VOC Ops",
    page_icon="📣",
    layout="wide",
)

pages = {
    "Overview": [
        st.Page("pages/dashboard.py", title="Dashboard", icon="📊", default=True),
    ],
    "Analyze": [
        st.Page("pages/voc_analysis.py", title="VOC Analyzer", icon="🔎"),
    ],
    "Design & Evaluation": [
        st.Page("pages/taxonomy.py", title="Taxonomy", icon="🧭"),
        st.Page("pages/evaluation.py", title="Evaluation", icon="🧪"),
    ],
}

navigation = st.navigation(pages)
st.sidebar.caption("AI VOC Ops · v0.1.0")
navigation.run()
