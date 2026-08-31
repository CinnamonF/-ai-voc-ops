"""Human review contracts for the provisional VOC evaluation seed.

The committed seed is an immutable source of synthetic examples.  This module
owns the small state machine used by the Evaluation page: edits can update a
working copy, but only :func:`approve_review_row` can promote a row to
``label_status=reviewed``.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd

from app.services.taxonomy import PRIORITIES, SENTIMENTS, TAXONOMY

REVIEW_COLUMNS = (
    "ticket_id",
    "customer_message",
    "category_gold",
    "subcategory_gold",
    "priority_gold",
    "sentiment_gold",
    "human_review_gold",
    "label_note",
    "dataset_subset",
    "source_type",
    "label_status",
    "label_version",
)

EDITABLE_REVIEW_COLUMNS = (
    "category_gold",
    "subcategory_gold",
    "priority_gold",
    "sentiment_gold",
    "human_review_gold",
    "label_note",
)

IMMUTABLE_SOURCE_COLUMNS = (
    "ticket_id",
    "customer_message",
    "dataset_subset",
    "source_type",
    "label_version",
)

ALLOWED_LABEL_STATUSES = frozenset({"provisional", "reviewed"})


class ReviewValidationError(ValueError):
    """Raised when a review-progress dataset violates its contract."""


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        result = pd.isna(value)
        return bool(result) if isinstance(result, (bool,)) else False
    except (TypeError, ValueError):
        return False


def _text(value: Any, *, field: str, row_number: int, allow_empty: bool = False) -> str:
    if _is_missing(value):
        raise ReviewValidationError(f"row {row_number}: {field} is required")
    result = str(value).strip()
    if not result and not allow_empty:
        raise ReviewValidationError(f"row {row_number}: {field} is required")
    return result


def parse_human_review_value(value: Any, *, field: str = "human_review_gold") -> bool:
    """Parse the supported boolean spellings without accepting arbitrary text."""

    if _is_missing(value):
        raise ReviewValidationError(f"{field} must be a boolean value")
    if isinstance(value, bool) or value.__class__.__name__ == "bool_":
        return bool(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value == 1:
            return True
        if value == 0:
            return False
        raise ReviewValidationError(f"{field} must be true/false, 1/0, yes/no, or y/n")
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise ReviewValidationError(f"{field} must be true/false, 1/0, yes/no, or y/n")


def _read_csv(source: Any) -> pd.DataFrame:
    try:
        if isinstance(source, (str, Path)):
            return pd.read_csv(source, dtype=str, keep_default_na=False)
        if isinstance(source, (bytes, bytearray)):
            return pd.read_csv(BytesIO(bytes(source)), dtype=str, keep_default_na=False)
        if hasattr(source, "getvalue"):
            return pd.read_csv(
                BytesIO(source.getvalue()), dtype=str, keep_default_na=False
            )
        if hasattr(source, "read"):
            payload = source.read()
            if isinstance(payload, str):
                payload = payload.encode("utf-8")
            return pd.read_csv(BytesIO(payload), dtype=str, keep_default_na=False)
    except (OSError, UnicodeDecodeError, pd.errors.ParserError, ValueError) as exc:
        raise ReviewValidationError(f"review progress CSV could not be read: {exc}") from exc
    raise ReviewValidationError("review progress source must be a CSV path, bytes, or file object")


def validate_review_dataset(
    frame: pd.DataFrame,
    *,
    seed: pd.DataFrame | None = None,
    require_full_seed: bool = False,
) -> pd.DataFrame:
    """Validate and safely normalize a review-progress frame.

    A normalized copy is returned; the caller's frame is never modified.  When
    ``seed`` is provided, all IDs and immutable source/provenance fields must
    correspond exactly to that seed.  ``require_full_seed`` additionally
    rejects missing seed rows, which is required for the full progress format.
    """

    if not isinstance(frame, pd.DataFrame):
        raise ReviewValidationError("review dataset must be a pandas DataFrame")
    if require_full_seed and seed is None:
        raise ReviewValidationError(
            "require_full_seed validation requires the original seed frame"
        )

    missing = [column for column in REVIEW_COLUMNS if column not in frame.columns]
    if missing:
        raise ReviewValidationError("missing required review columns: " + ", ".join(missing))
    unexpected = [column for column in frame.columns if column not in REVIEW_COLUMNS]
    if unexpected:
        raise ReviewValidationError("unexpected review columns: " + ", ".join(unexpected))
    if frame.empty:
        raise ReviewValidationError("review dataset must contain at least one row")

    validated = frame.loc[:, REVIEW_COLUMNS].copy()
    row_ids: list[str] = []
    seen: set[str] = set()
    for position, value in enumerate(validated["ticket_id"].tolist(), start=2):
        ticket_id = _text(value, field="ticket_id", row_number=position)
        if ticket_id in seen:
            raise ReviewValidationError(f"duplicate ticket_id: {ticket_id}")
        seen.add(ticket_id)
        row_ids.append(ticket_id)
    validated["ticket_id"] = row_ids

    for column in ("customer_message", "dataset_subset", "source_type", "label_version"):
        validated[column] = [
            _text(value, field=column, row_number=position)
            for position, value in enumerate(validated[column].tolist(), start=2)
        ]
    validated["label_note"] = [
        _text(value, field="label_note", row_number=position, allow_empty=True)
        for position, value in enumerate(validated["label_note"].tolist(), start=2)
    ]

    categories: list[str] = []
    subcategories: list[str] = []
    priorities: list[str] = []
    sentiments: list[str] = []
    review_flags: list[bool] = []
    statuses: list[str] = []
    for position, row in enumerate(validated.itertuples(index=False), start=2):
        category = _text(row.category_gold, field="category_gold", row_number=position)
        subcategory = _text(
            row.subcategory_gold, field="subcategory_gold", row_number=position
        )
        if category not in TAXONOMY:
            raise ReviewValidationError(
                f"row {position}: invalid category_gold: {category}"
            )
        if subcategory not in TAXONOMY[category]:
            raise ReviewValidationError(
                f"row {position}: invalid category/subcategory pair: "
                f"{category} > {subcategory}"
            )

        priority = _text(row.priority_gold, field="priority_gold", row_number=position).lower()
        if priority not in PRIORITIES:
            raise ReviewValidationError(
                f"row {position}: invalid priority_gold: {priority}"
            )
        sentiment = _text(
            row.sentiment_gold, field="sentiment_gold", row_number=position
        ).lower()
        if sentiment not in SENTIMENTS:
            raise ReviewValidationError(
                f"row {position}: invalid sentiment_gold: {sentiment}"
            )
        status = _text(row.label_status, field="label_status", row_number=position).lower()
        if status not in ALLOWED_LABEL_STATUSES:
            raise ReviewValidationError(
                f"row {position}: invalid label_status: {status}"
            )

        categories.append(category)
        subcategories.append(subcategory)
        priorities.append(priority)
        sentiments.append(sentiment)
        review_flags.append(parse_human_review_value(row.human_review_gold))
        statuses.append(status)

    validated["category_gold"] = categories
    validated["subcategory_gold"] = subcategories
    validated["priority_gold"] = priorities
    validated["sentiment_gold"] = sentiments
    validated["human_review_gold"] = review_flags
    validated["label_status"] = statuses

    if seed is not None:
        seed_validated = validate_review_dataset(seed)
        seed_by_id = seed_validated.set_index("ticket_id", drop=False)
        current_ids = set(validated["ticket_id"])
        seed_ids = set(seed_validated["ticket_id"])
        unknown = sorted(current_ids - seed_ids)
        if unknown:
            raise ReviewValidationError(
                "review progress contains unknown ticket_id(s): " + ", ".join(unknown)
            )
        if require_full_seed:
            missing_seed = sorted(seed_ids - current_ids)
            if missing_seed:
                raise ReviewValidationError(
                    "review progress is missing seed ticket_id(s): "
                    + ", ".join(missing_seed)
                )
        for position, row in validated.iterrows():
            seed_row = seed_by_id.loc[row["ticket_id"]]
            for column in IMMUTABLE_SOURCE_COLUMNS:
                if str(row[column]) != str(seed_row[column]):
                    raise ReviewValidationError(
                        f"source integrity mismatch for {row['ticket_id']}: {column}"
                    )

    return validated


def load_review_seed(path: str | Path) -> pd.DataFrame:
    """Load and validate the committed seed without changing it on disk."""

    frame = _read_csv(path)
    return validate_review_dataset(frame)


def load_review_progress(
    source: Any,
    *,
    seed: pd.DataFrame,
) -> pd.DataFrame:
    """Load a full review-progress CSV and validate it against ``seed``."""

    frame = _read_csv(source)
    return validate_review_dataset(frame, seed=seed, require_full_seed=True)


def export_review_progress(
    frame: pd.DataFrame,
    *,
    seed: pd.DataFrame | None = None,
) -> bytes:
    """Serialize a validated full progress frame as a UTF-8 CSV download."""

    validated = validate_review_dataset(
        frame,
        seed=seed,
        require_full_seed=seed is not None,
    )
    output = validated.copy()
    output["human_review_gold"] = output["human_review_gold"].map(
        lambda value: "true" if bool(value) else "false"
    )
    return output.loc[:, REVIEW_COLUMNS].to_csv(
        index=False, encoding="utf-8-sig"
    ).encode("utf-8-sig")


def review_counts(frame: pd.DataFrame) -> dict[str, int]:
    """Return review workflow counts, not model-performance metrics."""

    validated = validate_review_dataset(frame)
    statuses = validated["label_status"]
    reviewed = int(statuses.eq("reviewed").sum())
    provisional = int(statuses.eq("provisional").sum())
    return {
        "total": len(validated),
        "reviewed": reviewed,
        "provisional": provisional,
        "remaining": provisional,
    }


def _row_position(frame: pd.DataFrame, ticket_id: str) -> Any:
    matches = frame.index[frame["ticket_id"].eq(str(ticket_id))].tolist()
    if not matches:
        raise ReviewValidationError(f"ticket_id not found: {ticket_id}")
    return matches[0]


def edit_review_row(
    frame: pd.DataFrame,
    ticket_id: str,
    **updates: Any,
) -> pd.DataFrame:
    """Apply valid label edits while preserving the current review status."""

    unknown = sorted(set(updates) - set(EDITABLE_REVIEW_COLUMNS))
    if unknown:
        raise ReviewValidationError("unsupported review field(s): " + ", ".join(unknown))
    validated = validate_review_dataset(frame)
    position = _row_position(validated, ticket_id)
    edited = validated.copy()
    for field, value in updates.items():
        edited.at[position, field] = value
    # Validation is deliberately performed before returning, while status is
    # untouched.  A caller must invoke approve_review_row for promotion.
    return validate_review_dataset(edited)


def approve_review_row(
    frame: pd.DataFrame,
    ticket_id: str,
    **updates: Any,
) -> pd.DataFrame:
    """Explicitly approve one row after validating optional corrected labels."""

    edited = edit_review_row(frame, ticket_id, **updates)
    position = _row_position(edited, ticket_id)
    approved = edited.copy()
    approved.at[position, "label_status"] = "reviewed"
    return validate_review_dataset(approved)
