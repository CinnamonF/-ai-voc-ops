from pathlib import Path

import pandas as pd
import streamlit as st

from app.services.evaluation import EvaluationDataError, evaluate_results

ROOT = Path(__file__).resolve().parents[2]
SEED_PATH = ROOT / "evals" / "gold" / "voc_gold_seed_v0.2.csv"
RESULTS_DIR = ROOT / "evals" / "results"

st.title("Evaluation Lab")
st.caption(
    "Measure classification quality, operational risk detection, failure modes, "
    "and measured API usage with versioned evaluation data."
)

st.subheader("1. Evaluation dataset")
if not SEED_PATH.exists():
    st.error("v0.2 seed dataset을 찾을 수 없습니다.")
    st.stop()

gold = pd.read_csv(SEED_PATH)
status_counts = (
    gold["label_status"].astype(str).value_counts().rename_axis("label_status").reset_index(name="rows")
)
col1, col2, col3 = st.columns(3)
col1.metric("Seed rows", f"{len(gold):,}")
col2.metric("Subcategories covered", gold["subcategory_gold"].nunique())
col3.metric(
    "Human-reviewed rows",
    int(gold["label_status"].astype(str).str.lower().eq("reviewed").sum()),
)

st.warning(
    "현재 200건 seed는 synthetic + provisional입니다. 사람이 검수해 "
    "`label_status=reviewed`로 승인하기 전까지 여기서 나온 성능 수치는 "
    "포트폴리오 성과로 사용하면 안 됩니다."
)
st.dataframe(status_counts, width="stretch", hide_index=True)

coverage = (
    gold.groupby(["category_gold", "subcategory_gold"], as_index=False)
    .size()
    .rename(columns={"size": "rows"})
)
with st.expander("Dataset coverage", expanded=False):
    st.dataframe(coverage, width="stretch", hide_index=True)

st.divider()
st.subheader("2. Prediction results")
uploaded = st.file_uploader(
    "Gold evaluation runner가 생성한 *_predictions.csv 업로드",
    type=["csv"],
    key="evaluation_predictions",
)

results: pd.DataFrame | None = None
source_label = ""
if uploaded is not None:
    results = pd.read_csv(uploaded)
    source_label = uploaded.name
else:
    candidates = sorted(
        RESULTS_DIR.glob("*_predictions.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    ) if RESULTS_DIR.exists() else []
    if candidates:
        latest = candidates[0]
        st.caption(f"로컬 최신 결과 사용: {latest.name}")
        results = pd.read_csv(latest)
        source_label = latest.name

if results is None:
    st.info(
        "아직 평가 prediction이 없습니다. API 키를 설정한 뒤 "
        "`python -m evals.run_gold_eval`을 실행하세요. "
        "현재 seed는 provisional이므로 기본 실행은 prediction만 저장하고 "
        "publishable metric은 계산하지 않습니다."
    )
    st.stop()

include_provisional = st.checkbox(
    "Provisional synthetic labels로 exploratory metric 계산",
    value=False,
    help="개발 확인 전용입니다. 체크해서 나온 수치는 publishable 성능이 아닙니다.",
)

try:
    report = evaluate_results(results, include_provisional=include_provisional)
except EvaluationDataError as exc:
    st.info(str(exc))
    st.caption(
        "사람 검수 완료 행은 label_status를 reviewed로 바꿔 주세요. "
        "현재 구조를 시험하려면 위 exploratory 옵션을 사용할 수 있습니다."
    )
    st.stop()

if report.publishable:
    st.success(f"Reviewed labels 기반 평가 · {source_label}")
else:
    st.error(
        "EXPLORATORY / PROVISIONAL 평가입니다. "
        "아래 숫자를 README, 이력서, 포트폴리오 성과로 게시하지 마세요."
    )

st.subheader("3. Quality metrics")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Major Accuracy", f"{report.major['accuracy']:.1%}")
m2.metric("Major Macro F1", f"{report.major['macro_f1']:.1%}")
m3.metric("Subcategory Accuracy", f"{report.subcategory['accuracy']:.1%}")
m4.metric("Subcategory Macro F1", f"{report.subcategory['macro_f1']:.1%}")

r1, r2, r3, r4 = st.columns(4)
r1.metric("High-risk Recall", f"{report.high_risk['recall']:.1%}")
r2.metric("Human Review Precision", f"{report.human_review['precision']:.1%}")
r3.metric("Human Review Recall", f"{report.human_review['recall']:.1%}")
r4.metric("Evaluated rows", f"{report.rows_evaluated:,}")

st.subheader("4. Confusion matrices")
st.caption("행 = Gold label · 열 = AI prediction")
st.markdown("**Major category**")
st.dataframe(report.major_confusion, width="stretch")
with st.expander("Subcategory confusion matrix", expanded=False):
    st.dataframe(report.subcategory_confusion, width="stretch")

st.subheader("5. Per-subcategory quality")
per_subcategory = report.per_subcategory.sort_values(
    ["f1", "support"], ascending=[True, False]
)
st.dataframe(
    per_subcategory.style.format(
        {"precision": "{:.1%}", "recall": "{:.1%}", "f1": "{:.1%}"}
    ),
    width="stretch",
    hide_index=True,
)

st.subheader("6. Error Analysis")
if report.errors.empty:
    st.success("현재 평가 행에서 분류/우선순위/human-review 불일치가 없습니다.")
else:
    st.caption(
        "error_type은 사람이 검토하면서 수정할 수 있습니다. "
        "분류 원인을 기록한 뒤 CSV로 내려받아 experiment evidence로 보관하세요."
    )
    edited_errors = st.data_editor(
        report.errors,
        width="stretch",
        hide_index=True,
        column_config={
            "error_type": st.column_config.SelectboxColumn(
                "error_type",
                options=[
                    "",
                    "taxonomy_ambiguity",
                    "prompt_issue",
                    "customer_message_ambiguity",
                    "incorrect_gold_label",
                    "model_reasoning_issue",
                    "priority_policy",
                    "human_review_policy",
                ],
            ),
        },
        disabled=[column for column in report.errors.columns if column != "error_type"],
    )
    st.download_button(
        "Download annotated errors",
        data=edited_errors.to_csv(index=False).encode("utf-8-sig"),
        file_name="voc_error_analysis.csv",
        mime="text/csv",
    )

st.subheader("7. Run provenance & measured usage")
v1, v2, v3 = st.columns(3)
v1.metric("Dataset version", report.versions["dataset"])
v2.metric("Prompt version", report.versions["prompt"])
v3.metric("Taxonomy version", report.versions["taxonomy"])

usage = pd.DataFrame(
    [
        {"field": "models", "value": ", ".join(report.usage["models"]) or "unknown"},
        {"field": "input_tokens", "value": report.usage["input_tokens"]},
        {"field": "cached_input_tokens", "value": report.usage["cached_input_tokens"]},
        {"field": "output_tokens", "value": report.usage["output_tokens"]},
        {
            "field": "estimated_cost_usd",
            "value": (
                report.usage["estimated_cost_usd"]
                if report.usage["estimated_cost_usd"] is not None
                else "not configured"
            ),
        },
    ]
)
st.dataframe(usage, width="stretch", hide_index=True)
st.caption(
    "Token counts are measured API usage. Cost appears only when dated operator-supplied "
    "prices were configured for the run."
)
