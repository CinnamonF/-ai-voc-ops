"""Compatibility exports derived from the canonical service taxonomy."""

from app.services.taxonomy import (
    HUMAN_REVIEW_RULES,
    PRIORITY_RULES,
    TAXONOMY_DETAILS,
)

TAXONOMY = {
    category: [dict(item) for item in items]
    for category, items in TAXONOMY_DETAILS.items()
}
