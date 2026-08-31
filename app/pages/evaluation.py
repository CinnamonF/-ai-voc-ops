from hashlib import sha256
from pathlib import Path

import pandas as pd
import streamlit as st

from app.services.evaluation import EvaluationDataError, evaluate_results
from app.services.evaluation_review import (
    ReviewValidationError,
    approve_review_row,
    export_review_progress,
    load_review_progress,
    load_review_seed,
    review_counts,
    validate_review_dataset,
)
from app.services.taxonomy import PRIORITIES, SENTIMENTS, TAXONOMY

ROOT = Path(__file__).resolve().parents[2]
SEED_PATH = ROOT / "evals" / "gold" / "voc_gold_seed_v0.2.csv"
RESULTS_DIR = ROOT / "evals" / "results"
REVIEW_FRAME_KEY = "evaluation_review_progress_frame"
REVIEW_IMPORT_DIGEST_KEY = "evaluation_review_import_digest"
REVIEW_ACTIVE_TICKET_KEY = "evaluation_review_active_ticket"


def _status_counts(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame["label_status"]
        .astype(str)
        .str.lower()
        .value_counts()
        .rename_axis("label_status")
        .reset_index(name="rows")
    )


def _next_provisional_ticket(frame: pd.DataFrame, current_ticket: str) -> str | None:
    provisional = frame.loc[
        frame["label_status"].astype(str).str.lower().eq("provisional"), "ticket_id"
    ].astype(str).tolist()
    if not provisional:
        return None
    if current_ticket in provisional:
        start = provisional.index(current_ticket)
        return provisional[(start + 1) % len(provisional)]
    return provisional[0]


st.title("Evaluation Lab")
st.caption(
    "Measure classification quality, operational risk detection, failure modes, "
    "and measured API usage with versioned evaluation data."
)

st.subheader("1. Evaluation dataset")
if not SEED_PATH.exists():
    st.error("v0.2 seed dataset을 찾을 수 없습니다.")
    st.stop()

try:
    gold = load_review_seed(SEED_PATH)
except ReviewValidationError as exc:
    st.error(f"v0.2 seed validation failed: {exc}")
    st.stop()

if REVIEW_FRAME_KEY not in st.session_state:
    st.session_state[REVIEW_FRAME_KEY] = gold.copy()
working = st.session_state[REVIEW_FRAME_KEY]
try:
    # Keep the session working copy safe even when Streamlit reruns the page.
    working = validate_review_dataset(working, seed=gold, require_full_seed=True)
    st.session_state[REVIEW_FRAME_KEY] = working
except ReviewValidationError as exc:
    st.error(f"Review working copy is invalid: {exc}")
    st.session_state[REVIEW_FRAME_KEY] = gold.copy()
    working = st.session_state[REVIEW_FRAME_KEY]

counts = review_counts(working)
col1, col2, col3 = st.columns(3)
col1.metric("Seed rows", f"{len(gold):,}")
col2.metric("Subcategories covered", gold["subcategory_gold"].nunique())
col3.metric("Reviewed in working copy", f"{counts['reviewed']:,}")

st.warning(
    "원본 200건 seed는 synthetic + provisional로 보존됩니다. 사람이 검수해 "
    "명시적으로 승인한 working-copy 행만 `label_status=reviewed`가 되며, "
    "그 전까지 성능 수치는 포트폴리오 성과로 사용할 수 없습니다."
)
st.dataframe(_status_counts(working), width="stretch", hide_index=True)

coverage = (
    gold.groupby(["category_gold", "subcategory_gold"], as_index=False)
    .size()
    .rename(columns={"size": "rows"})
)
with st.expander("Dataset coverage", expanded=False):
    st.dataframe(coverage, width="stretch", hide_index=True)

st.divider()
st.subheader("2. Human review workflow")
st.caption(
    "기존 라벨은 provisional 제안입니다. 행을 열거나 수정하는 것만으로는 저장·승인되지 않으며, "
    "`Approve & mark reviewed`를 눌렀을 때만 해당 행이 reviewed가 됩니다. "
    "ticket_id와 customer_message는 원본 seed와 대조됩니다."
)

imported_review = st.file_uploader(
    "Resume from a review-progress CSV",
    type=["csv"],
    key="evaluation_review_import",
    help="The file must contain every seed ticket and preserve immutable source fields.",
)
if imported_review is not None:
    payload = imported_review.getvalue()
    digest = sha256(payload).hexdigest()
    if digest != st.session_state.get(REVIEW_IMPORT_DIGEST_KEY):
        try:
            imported_frame = load_review_progress(payload, seed=gold)
        except ReviewValidationError as exc:
            st.error(f"Review progress rejected: {exc}")
        else:
            st.session_state[REVIEW_FRAME_KEY] = imported_frame
            st.session_state[REVIEW_IMPORT_DIGEST_KEY] = digest
            working = imported_frame
            counts = review_counts(working)
            st.success(
                f"Review progress loaded: {counts['reviewed']} reviewed / {counts['total']} total."
            )
    else:
        working = st.session_state[REVIEW_FRAME_KEY]
        counts = review_counts(working)

st.download_button(
    "Export full review progress",
    data=export_review_progress(working, seed=gold),
    file_name="voc_gold_review_progress.csv",
    mime="text/csv",
    key="evaluation_review_export",
    help="Includes reviewed and remaining provisional rows; the committed seed is not overwritten.",
)

progress_col1, progress_col2, progress_col3, progress_col4 = st.columns(4)
progress_col1.metric("Reviewed", f"{counts['reviewed']} / {counts['total']}")
progress_col2.metric("Remaining", counts["remaining"])
progress_col3.metric("Provisional", counts["provisional"])
progress_col4.metric("Reviewed status", counts["reviewed"])

if counts["reviewed"] == 0:
    st.info(
        "No human-reviewed labels are available for publishable evaluation yet. "
        "Review and explicitly approve rows above; this progress is not model performance."
    )

filter_col1, filter_col2, filter_col3 = st.columns(3)
status_filter = filter_col1.radio(
    "Status filter",
    ["All", "Provisional only", "Reviewed only"],
    index=1,
    horizontal=True,
    key="evaluation_review_status_filter",
)
category_filter = filter_col2.selectbox(
    "Category filter",
    ["All", *TAXONOMY.keys()],
    key="evaluation_review_category_filter",
)
subcategory_filter = filter_col3.selectbox(
    "Subcategory filter",
    ["All", *sorted(working["subcategory_gold"].astype(str).unique())],
    key="evaluation_review_subcategory_filter",
)

filtered = working.copy()
if status_filter == "Provisional only":
    filtered = filtered[filtered["label_status"].eq("provisional")]
elif status_filter == "Reviewed only":
    filtered = filtered[filtered["label_status"].eq("reviewed")]
if category_filter != "All":
    filtered = filtered[filtered["category_gold"].eq(category_filter)]
if subcategory_filter != "All":
    filtered = filtered[filtered["subcategory_gold"].eq(subcategory_filter)]

visible_ids = filtered["ticket_id"].astype(str).tolist()
if not visible_ids:
    st.info("현재 필터에 해당하는 검수 행이 없습니다.")
else:
    active_ticket = st.session_state.get(REVIEW_ACTIVE_TICKET_KEY)
    if active_ticket not in visible_ids:
        active_ticket = visible_ids[0]
        st.session_state[REVIEW_ACTIVE_TICKET_KEY] = active_ticket
    # Include the current ticket in the widget key. Navigation changes the
    # separate active-ticket state before rerun, so the next row gets a fresh
    # selectbox without mutating an already-instantiated widget.
    active_widget_key = f"evaluation_review_active_ticket_widget_{active_ticket}"
    active_ticket = st.selectbox(
        "Active ticket",
        visible_ids,
        index=visible_ids.index(active_ticket),
        key=active_widget_key,
    )
    st.session_state[REVIEW_ACTIVE_TICKET_KEY] = active_ticket
    row = working.loc[working["ticket_id"].eq(active_ticket)].iloc[0]
    status = str(row["label_status"]).upper()
    if status == "REVIEWED":
        st.success("REVIEWED · 이 행은 명시적으로 승인되었습니다.")
    else:
        st.warning("PROVISIONAL · 기존 라벨은 사람 확인 전 제안이며 아직 publishable하지 않습니다.")
    st.markdown(f"**Customer message**  \n{row['customer_message']}")

    details = pd.DataFrame(
        [
            {
                "ticket_id": row["ticket_id"],
                "category_gold": row["category_gold"],
                "subcategory_gold": row["subcategory_gold"],
                "priority_gold": row["priority_gold"],
                "sentiment_gold": row["sentiment_gold"],
                "human_review_gold": bool(row["human_review_gold"]),
                "label_note": row["label_note"],
                "dataset_subset": row["dataset_subset"],
                "source_type": row["source_type"],
                "label_status": row["label_status"],
                "label_version": row["label_version"],
            }
        ]
    )
    st.dataframe(details, width="stretch", hide_index=True)

    nav_col1, nav_col2, nav_col3 = st.columns(3)
    active_index = visible_ids.index(active_ticket)
    if nav_col1.button(
        "Previous",
        disabled=len(visible_ids) < 2,
        key=f"evaluation_review_previous_{active_ticket}",
    ):
        st.session_state[REVIEW_ACTIVE_TICKET_KEY] = visible_ids[
            (active_index - 1) % len(visible_ids)
        ]
        st.rerun()
    if nav_col2.button(
        "Next",
        disabled=len(visible_ids) < 2,
        key=f"evaluation_review_next_{active_ticket}",
    ):
        st.session_state[REVIEW_ACTIVE_TICKET_KEY] = visible_ids[
            (active_index + 1) % len(visible_ids)
        ]
        st.rerun()
    next_provisional = _next_provisional_ticket(working, active_ticket)
    if nav_col3.button(
        "Next provisional",
        disabled=next_provisional is None,
        key=f"evaluation_review_next_provisional_{active_ticket}",
    ):
        st.session_state[REVIEW_ACTIVE_TICKET_KEY] = next_provisional
        st.rerun()

    if status == "PROVISIONAL":
        edit_col1, edit_col2 = st.columns(2)
        category_value = edit_col1.selectbox(
            "Category",
            list(TAXONOMY.keys()),
            index=list(TAXONOMY.keys()).index(str(row["category_gold"])),
            key=f"evaluation_review_category_{active_ticket}",
        )
        subcategories = list(TAXONOMY[category_value])
        current_subcategory = str(row["subcategory_gold"])
        subcategory_value = edit_col2.selectbox(
            "Subcategory",
            subcategories,
            index=(
                subcategories.index(current_subcategory)
                if current_subcategory in subcategories
                else 0
            ),
            key=f"evaluation_review_subcategory_{active_ticket}_{category_value}",
        )
        edit_col3, edit_col4 = st.columns(2)
        priority_value = edit_col3.selectbox(
            "Priority",
            list(PRIORITIES),
            index=list(PRIORITIES).index(str(row["priority_gold"])),
            key=f"evaluation_review_priority_{active_ticket}",
        )
        sentiment_value = edit_col4.selectbox(
            "Sentiment",
            list(SENTIMENTS),
            index=list(SENTIMENTS).index(str(row["sentiment_gold"])),
            key=f"evaluation_review_sentiment_{active_ticket}",
        )
        human_review_value = st.checkbox(
            "Human review required",
            value=bool(row["human_review_gold"]),
            key=f"evaluation_review_human_{active_ticket}",
        )
        label_note_value = st.text_area(
            "Human review note (optional)",
            value=str(row["label_note"]),
            key=f"evaluation_review_note_{active_ticket}",
        )
        if st.button(
            "Approve & mark reviewed",
            type="primary",
            key=f"evaluation_review_approve_{active_ticket}",
        ):
            try:
                approved = approve_review_row(
                    working,
                    active_ticket,
                    category_gold=category_value,
                    subcategory_gold=subcategory_value,
                    priority_gold=priority_value,
                    sentiment_gold=sentiment_value,
                    human_review_gold=human_review_value,
                    label_note=label_note_value,
                )
            except ReviewValidationError as exc:
                st.error(f"Approval rejected: {exc}")
            else:
                st.session_state[REVIEW_FRAME_KEY] = approved
                st.session_state[REVIEW_ACTIVE_TICKET_KEY] = (
                    _next_provisional_ticket(approved, active_ticket)
                    or active_ticket
                )
                st.rerun()
    else:
        st.caption("Reviewed rows are read-only in this workflow. Start with provisional rows for new approvals.")

st.divider()
st.subheader("3. Prediction results")
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
    candidates = (
        sorted(
            RESULTS_DIR.glob("*_predictions.csv"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if RESULTS_DIR.exists()
        else []
    )
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

st.subheader("4. Quality metrics")
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

st.subheader("5. Confusion matrices")
st.caption("행 = Gold label · 열 = AI prediction")
st.markdown("**Major category**")
st.dataframe(report.major_confusion, width="stretch")
with st.expander("Subcategory confusion matrix", expanded=False):
    st.dataframe(report.subcategory_confusion, width="stretch")

st.subheader("6. Per-subcategory quality")
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

st.subheader("7. Error Analysis")
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

st.subheader("8. Run provenance & measured usage")
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
