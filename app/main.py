import streamlit as st

from app.utils.config import VERSION

st.set_page_config(
    page_title="AI VOC Ops",
    page_icon="📣",
    layout="wide",
)

pages = {
    "Overview": [
        st.Page("pages/dashboard.py", title="Dashboard", icon="📊", default=True),
    ],
    "Pilot": [
        st.Page("pages/pilot_test.py", title="Pilot Test", icon="🧑‍💻"),
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
st.sidebar.caption(f"AI VOC Ops · v{VERSION}")
navigation.run()
