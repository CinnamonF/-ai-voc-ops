from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from app.services.evaluation import evaluate_results
from app.services.evaluation_review import (
    ReviewValidationError,
    approve_review_row,
    edit_review_row,
    export_review_progress,
    load_review_progress,
    load_review_seed,
    review_counts,
    validate_review_dataset,
)

SEED_PATH = Path(__file__).resolve().parents[1] / "evals" / "gold" / "voc_gold_seed_v0.2.csv"
PAGE_PATH = Path(__file__).resolve().parents[1] / "app" / "pages" / "evaluation.py"


@pytest.fixture
def seed() -> pd.DataFrame:
    return load_review_seed(SEED_PATH)


def test_valid_seed_review_schema_and_counts(seed: pd.DataFrame):
    validated = validate_review_dataset(seed)
    assert validated.shape == (200, 12)
    assert review_counts(validated) == {
        "total": 200,
        "reviewed": 0,
        "provisional": 200,
        "remaining": 200,
    }
    assert validated["human_review_gold"].dtype == bool


def test_full_seed_validation_requires_a_seed_reference(seed: pd.DataFrame):
    with pytest.raises(ReviewValidationError, match="requires the original seed"):
        validate_review_dataset(seed, require_full_seed=True)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("category_gold", "not-a-category", "invalid category"),
        ("subcategory_gold", "중복 결제", "invalid category/subcategory pair"),
        ("priority_gold", "urgent", "invalid priority"),
        ("sentiment_gold", "angry", "invalid sentiment"),
        ("label_status", "approved", "invalid label_status"),
    ],
)
def test_invalid_review_values_are_rejected(
    seed: pd.DataFrame, column: str, value: str, message: str
):
    invalid = seed.copy()
    invalid.loc[0, column] = value
    with pytest.raises(ReviewValidationError, match=message):
        validate_review_dataset(invalid)


def test_duplicate_ticket_id_is_rejected(seed: pd.DataFrame):
    invalid = seed.copy()
    invalid.loc[1, "ticket_id"] = invalid.loc[0, "ticket_id"]
    with pytest.raises(ReviewValidationError, match="duplicate ticket_id"):
        validate_review_dataset(invalid)


def test_invalid_human_review_value_and_unexpected_column_are_rejected(seed: pd.DataFrame):
    invalid_boolean = seed.copy()
    invalid_boolean["human_review_gold"] = invalid_boolean["human_review_gold"].astype(object)
    invalid_boolean.loc[0, "human_review_gold"] = "sometimes"
    with pytest.raises(ReviewValidationError, match="human_review_gold"):
        validate_review_dataset(invalid_boolean)

    malformed = seed.copy()
    malformed["unexpected"] = "not part of the review schema"
    with pytest.raises(ReviewValidationError, match="unexpected review columns"):
        load_review_progress(malformed.to_csv(index=False).encode("utf-8"), seed=seed)


def test_open_edit_and_navigation_do_not_review_rows(seed: pd.DataFrame):
    ticket_id = seed.loc[0, "ticket_id"]
    # Opening is a read-only copy; a label edit is also explicitly status-neutral.
    opened = seed.copy()
    assert opened.loc[0, "label_status"] == "provisional"
    edited = edit_review_row(opened, ticket_id, label_note="검토 중")
    assert edited.loc[0, "label_note"] == "검토 중"
    assert edited.loc[0, "label_status"] == "provisional"
    # A caller can select another row without mutating either row's status.
    navigated = edited.iloc[[1, 0]].reset_index(drop=True)
    assert set(navigated["label_status"]) == {"provisional"}
    assert set(seed["label_status"]) == {"provisional"}


def test_only_explicit_approval_promotes_a_row(seed: pd.DataFrame):
    ticket_id = seed.loc[0, "ticket_id"]
    approved = approve_review_row(seed, ticket_id)
    assert approved.loc[0, "label_status"] == "reviewed"
    assert int(approved["label_status"].eq("reviewed").sum()) == 1
    assert seed.loc[0, "label_status"] == "provisional"


def test_corrected_labels_persist_only_with_explicit_approval(seed: pd.DataFrame):
    ticket_id = seed.loc[0, "ticket_id"]
    corrected = approve_review_row(
        seed,
        ticket_id,
        category_gold="교환/반품",
        subcategory_gold="파손",
        priority_gold="critical",
        sentiment_gold="negative",
        human_review_gold=True,
        label_note="기능 문제가 아니라 물리적 파손으로 판단",
    )
    row = corrected.loc[corrected["ticket_id"].eq(ticket_id)].iloc[0]
    assert row["category_gold"] == "교환/반품"
    assert row["subcategory_gold"] == "파손"
    assert row["priority_gold"] == "critical"
    assert row["sentiment_gold"] == "negative"
    assert bool(row["human_review_gold"]) is True
    assert row["label_note"] == "기능 문제가 아니라 물리적 파손으로 판단"
    assert row["label_status"] == "reviewed"
    assert int(corrected["label_status"].eq("provisional").sum()) == 199


def test_export_import_round_trip_preserves_review_progress(seed: pd.DataFrame):
    progress = approve_review_row(seed, seed.loc[0, "ticket_id"], label_note="승인")
    payload = export_review_progress(progress, seed=seed)
    restored = load_review_progress(payload, seed=seed)
    assert restored.loc[0, "label_status"] == "reviewed"
    assert restored.loc[0, "label_note"] == "승인"
    assert int(restored["label_status"].eq("provisional").sum()) == 199
    assert restored["ticket_id"].tolist() == progress["ticket_id"].tolist()


def test_import_rejects_missing_column(seed: pd.DataFrame):
    malformed = seed.drop(columns=["label_note"])
    payload = malformed.to_csv(index=False).encode("utf-8")
    with pytest.raises(ReviewValidationError, match="missing required review columns"):
        load_review_progress(payload, seed=seed)


def test_import_rejects_unknown_and_missing_ticket_ids(seed: pd.DataFrame):
    unknown = seed.copy()
    unknown.loc[0, "ticket_id"] = "UNKNOWN-0001"
    with pytest.raises(ReviewValidationError, match="unknown ticket_id"):
        load_review_progress(unknown.to_csv(index=False).encode("utf-8"), seed=seed)

    missing = seed.iloc[:-1].copy()
    with pytest.raises(ReviewValidationError, match="missing seed ticket_id"):
        load_review_progress(missing.to_csv(index=False).encode("utf-8"), seed=seed)


@pytest.mark.parametrize("column", ["customer_message", "source_type", "label_version"])
def test_import_rejects_source_tampering(seed: pd.DataFrame, column: str):
    tampered = seed.copy()
    tampered.loc[0, column] = "tampered"
    with pytest.raises(ReviewValidationError, match="source integrity mismatch"):
        load_review_progress(tampered.to_csv(index=False).encode("utf-8"), seed=seed)


def test_reviewed_only_publishability_gate_with_mixed_rows(seed: pd.DataFrame):
    results = seed.iloc[:2].copy()
    results.loc[results.index[0], "label_status"] = "reviewed"
    for column in ("category", "subcategory", "priority", "sentiment"):
        results[column] = results[f"{column}_gold"]
    results["requires_human_review"] = results["human_review_gold"]
    results["analysis_status"] = "success"
    report = evaluate_results(results)
    assert report.rows_total == 2
    assert report.rows_evaluated == 1
    assert report.publishable is True


def _app() -> AppTest:
    return AppTest.from_file(str(PAGE_PATH)).run(timeout=30)


def test_app_test_approve_is_explicit_and_zero_review_state_is_truthful():
    app = _app()
    assert not app.exception
    assert any("No human-reviewed labels" in item.value for item in app.info)
    frame = app.session_state["evaluation_review_progress_frame"]
    assert frame.loc[0, "label_status"] == "provisional"

    app.button(key="evaluation_review_approve_GOLD-0001").click().run(timeout=30)
    assert not app.exception
    frame = app.session_state["evaluation_review_progress_frame"]
    assert frame.loc[0, "label_status"] == "reviewed"
    assert frame["label_status"].eq("reviewed").sum() == 1


def test_app_test_corrected_labels_require_approval():
    app = _app()
    app.selectbox(key="evaluation_review_active_ticket_widget_GOLD-0001").set_value(
        "GOLD-0002"
    ).run(timeout=30)
    app.selectbox(key="evaluation_review_category_GOLD-0002").set_value(
        "교환/반품"
    ).run(timeout=30)
    app.selectbox(
        key="evaluation_review_subcategory_GOLD-0002_교환/반품"
    ).set_value("파손").run(timeout=30)
    app.selectbox(key="evaluation_review_priority_GOLD-0002").set_value(
        "critical"
    ).run(timeout=30)
    app.text_area(key="evaluation_review_note_GOLD-0002").set_value(
        "기능 문제가 아니라 물리적 파손으로 판단"
    ).run(timeout=30)
    frame = app.session_state["evaluation_review_progress_frame"]
    assert frame.loc[1, "label_status"] == "provisional"

    app.button(key="evaluation_review_approve_GOLD-0002").click().run(timeout=30)
    assert not app.exception
    row = app.session_state["evaluation_review_progress_frame"].iloc[1]
    assert row["category_gold"] == "교환/반품"
    assert row["subcategory_gold"] == "파손"
    assert row["priority_gold"] == "critical"
    assert row["label_status"] == "reviewed"


def test_app_test_navigation_does_not_approve():
    app = _app()
    app.button(key="evaluation_review_next_GOLD-0001").click().run(timeout=30)
    assert app.session_state["evaluation_review_active_ticket"] == "GOLD-0002"
    assert set(app.session_state["evaluation_review_progress_frame"]["label_status"]) == {
        "provisional"
    }
    app.button(key="evaluation_review_previous_GOLD-0002").click().run(timeout=30)
    assert app.session_state["evaluation_review_active_ticket"] == "GOLD-0001"
    assert set(app.session_state["evaluation_review_progress_frame"]["label_status"]) == {
        "provisional"
    }


def test_app_test_import_resume_preserves_reviewed_rows(seed: pd.DataFrame):
    progress = approve_review_row(
        seed,
        seed.loc[0, "ticket_id"],
        label_note="승인된 검수 메모",
    )
    progress = approve_review_row(
        progress,
        seed.loc[1, "ticket_id"],
        category_gold="교환/반품",
        subcategory_gold="파손",
        priority_gold="critical",
        sentiment_gold="negative",
        human_review_gold=True,
        label_note="수정된 검수 메모",
    )
    payload = export_review_progress(progress, seed=seed)
    app = _app()
    app.file_uploader(key="evaluation_review_import").set_value(
        ("voc_gold_review_progress.csv", payload, "text/csv")
    ).run(timeout=30)
    assert not app.exception
    restored = app.session_state["evaluation_review_progress_frame"]
    assert restored.loc[0, "label_status"] == "reviewed"
    assert restored.loc[0, "label_note"] == "승인된 검수 메모"
    assert restored.loc[1, "category_gold"] == "교환/반품"
    assert restored.loc[1, "subcategory_gold"] == "파손"
    assert restored.loc[1, "priority_gold"] == "critical"
    assert restored.loc[1, "label_status"] == "reviewed"
    assert restored["label_status"].eq("provisional").sum() == 198
