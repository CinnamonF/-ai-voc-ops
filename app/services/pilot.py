"""Pilot-testing utilities for safe feedback collection and persistence."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any
from urllib import error, request
from uuid import uuid4

from app.services.classifier import VOCAnalysis
from app.services.taxonomy import PRIORITIES, SENTIMENTS, TAXONOMY
from app.utils.config import PROMPT_VERSION, TAXONOMY_VERSION

DEFAULT_BATCH_ROW_LIMIT = 100
DEFAULT_TEXT_CHAR_LIMIT = 4000
DEFAULT_SINGLE_ANALYSIS_LIMIT = 20
DEFAULT_FEEDBACK_TABLE = "pilot_feedback"
MAX_CONFIGURED_BATCH_ROWS = 500

_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_PHONE_RE = re.compile(
    r"(?<!\d)(?:\+?82[-.\s]?)?(?:0?1[016789])(?:[-.\s]?\d){7,8}(?!\d)"
)
_LONG_NUMBER_RE = re.compile(r"(?<!\d)\d{7,}(?!\d)")
_TABLE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class PilotConfigurationError(RuntimeError):
    """Raised when pilot/deployment configuration is invalid."""


class FeedbackPersistenceError(RuntimeError):
    """Raised when feedback cannot be written to the configured backend."""


@dataclass(frozen=True)
class PilotFeedback:
    feedback_id: str
    created_at: str
    session_id: str
    message_redacted: str
    message_fingerprint: str
    prediction_category: str
    prediction_subcategory: str
    prediction_priority: str
    prediction_sentiment: str
    prediction_human_review: bool
    prediction_reason: str
    is_correct: bool
    corrected_category: str
    corrected_subcategory: str
    corrected_priority: str
    corrected_sentiment: str
    corrected_human_review: bool
    feedback_note: str
    model: str | None
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    estimated_cost_usd: float | None
    prompt_version: str
    taxonomy_version: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _positive_int_env(name: str, default: int, *, maximum: int | None = None) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise PilotConfigurationError(f"{name}은(는) 양의 정수여야 합니다.") from exc
    if value < 1 or (maximum is not None and value > maximum):
        suffix = f"1~{maximum}" if maximum is not None else "1 이상"
        raise PilotConfigurationError(f"{name}은(는) {suffix} 범위여야 합니다.")
    return value


def get_batch_row_limit() -> int:
    return _positive_int_env(
        "PILOT_MAX_BATCH_ROWS",
        DEFAULT_BATCH_ROW_LIMIT,
        maximum=MAX_CONFIGURED_BATCH_ROWS,
    )


def get_text_char_limit() -> int:
    return _positive_int_env("PILOT_MAX_TEXT_CHARS", DEFAULT_TEXT_CHAR_LIMIT, maximum=10000)


def get_single_analysis_limit() -> int:
    return _positive_int_env(
        "PILOT_MAX_SINGLE_ANALYSES_PER_SESSION",
        DEFAULT_SINGLE_ANALYSIS_LIMIT,
        maximum=200,
    )


def redact_pii(text: str) -> str:
    """Redact common contact/identifier patterns before feedback persistence."""
    redacted = str(text)
    redacted = _EMAIL_RE.sub("[EMAIL]", redacted)
    redacted = _PHONE_RE.sub("[PHONE]", redacted)
    redacted = _LONG_NUMBER_RE.sub("[NUMBER]", redacted)
    return redacted


def _validate_corrected_labels(
    *,
    category: str,
    subcategory: str,
    priority: str,
    sentiment: str,
) -> None:
    if category not in TAXONOMY:
        raise ValueError(f"알 수 없는 정답 대분류: {category}")
    if subcategory not in TAXONOMY[category]:
        raise ValueError(f"정답 taxonomy 조합이 올바르지 않습니다: {category} > {subcategory}")
    if priority not in PRIORITIES:
        raise ValueError(f"알 수 없는 정답 우선순위: {priority}")
    if sentiment not in SENTIMENTS:
        raise ValueError(f"알 수 없는 정답 감정 라벨: {sentiment}")


def build_feedback_record(
    *,
    session_id: str,
    message: str,
    analysis: VOCAnalysis,
    is_correct: bool,
    corrected_category: str,
    corrected_subcategory: str,
    corrected_priority: str,
    corrected_sentiment: str,
    corrected_human_review: bool,
    feedback_note: str = "",
) -> PilotFeedback:
    """Build a privacy-reduced feedback record from one model result."""
    classification = analysis.classification
    _validate_corrected_labels(
        category=corrected_category,
        subcategory=corrected_subcategory,
        priority=corrected_priority,
        sentiment=corrected_sentiment,
    )
    redacted = redact_pii(message).strip()
    fingerprint = hashlib.sha256(redacted.encode("utf-8")).hexdigest()[:20]
    usage = analysis.usage

    return PilotFeedback(
        feedback_id=str(uuid4()),
        created_at=datetime.now(timezone.utc).isoformat(),
        session_id=session_id,
        message_redacted=redacted,
        message_fingerprint=fingerprint,
        prediction_category=str(classification["category"]),
        prediction_subcategory=str(classification["subcategory"]),
        prediction_priority=str(classification["priority"]),
        prediction_sentiment=str(classification["sentiment"]),
        prediction_human_review=bool(classification["requires_human_review"]),
        prediction_reason=str(classification["reason"]),
        is_correct=bool(is_correct),
        corrected_category=corrected_category,
        corrected_subcategory=corrected_subcategory,
        corrected_priority=corrected_priority,
        corrected_sentiment=corrected_sentiment,
        corrected_human_review=bool(corrected_human_review),
        feedback_note=str(feedback_note).strip()[:1000],
        model=usage.model,
        input_tokens=usage.input_tokens,
        cached_input_tokens=usage.cached_input_tokens,
        output_tokens=usage.output_tokens,
        estimated_cost_usd=usage.estimated_cost_usd,
        prompt_version=PROMPT_VERSION,
        taxonomy_version=TAXONOMY_VERSION,
    )


def _feedback_config() -> tuple[str, str, str] | None:
    url = os.getenv("SUPABASE_URL", "").strip().rstrip("/")
    key = os.getenv("SUPABASE_ANON_KEY", "").strip()
    table = os.getenv("SUPABASE_FEEDBACK_TABLE", DEFAULT_FEEDBACK_TABLE).strip()
    if not url and not key:
        return None
    if not url or not key:
        raise PilotConfigurationError(
            "Supabase 피드백 저장을 사용하려면 SUPABASE_URL과 SUPABASE_ANON_KEY를 모두 설정하세요."
        )
    if not _TABLE_RE.fullmatch(table):
        raise PilotConfigurationError("SUPABASE_FEEDBACK_TABLE 이름이 올바르지 않습니다.")
    return url, key, table


def is_feedback_store_configured() -> bool:
    try:
        return _feedback_config() is not None
    except PilotConfigurationError:
        return False


def save_feedback(record: PilotFeedback, *, timeout_seconds: float = 8.0) -> None:
    """Persist one feedback row through Supabase REST when configured."""
    config = _feedback_config()
    if config is None:
        raise FeedbackPersistenceError("영구 피드백 저장소가 설정되어 있지 않습니다.")
    url, key, table = config
    endpoint = f"{url}/rest/v1/{table}"
    payload = json.dumps(record.as_dict(), ensure_ascii=False).encode("utf-8")
    req = request.Request(
        endpoint,
        data=payload,
        method="POST",
        headers={
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal",
        },
    )
    try:
        with request.urlopen(req, timeout=timeout_seconds) as response:
            if response.status not in {200, 201, 204}:
                raise FeedbackPersistenceError(
                    f"피드백 저장소가 HTTP {response.status}를 반환했습니다."
                )
    except error.HTTPError as exc:
        raise FeedbackPersistenceError(
            f"피드백을 저장하지 못했습니다 (HTTP {exc.code})."
        ) from exc
    except error.URLError as exc:
        raise FeedbackPersistenceError("피드백 저장소에 연결하지 못했습니다.") from exc
