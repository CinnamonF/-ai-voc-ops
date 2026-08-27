# Decision Log

## 2026-08-27 — Project direction

The project will focus on CX Operations and VOC intelligence rather than building a generic customer-support chatbot.

Reason:
- Better alignment with CS/VOC operations roles
- Demonstrates operational problem solving, not only LLM usage
- Makes evaluation and measurable improvement possible

## 2026-08-28 — Canonical taxonomy and two validation layers

`app/services/taxonomy.py` is the canonical code definition. Prompt labels, JSON Schema enums, compatibility exports, and the Streamlit Taxonomy page derive from it.

Strict Structured Outputs remain the first validation layer. Application code then validates exact fields, enum values, non-empty reasons, and the selected category/subcategory pair. JSON Schema cannot express the project-specific parent-child rule clearly enough on its own.

## 2026-08-28 — Human review is an operational safety rule

The model may recommend human review, but application code forces it to `true` for `high` or `critical` priority and the configured high-risk subcategories. Negative sentiment alone does not trigger review. This separates customer emotion from operational risk.

## 2026-08-28 — Batch failures remain visible and local

Each CSV row is classified independently. A failed request preserves the source row, records `analysis_status=failed`, adds a safe `analysis_error`, and leaves classification fields empty. Failed rows are excluded from classification charts and human-review/high-priority counts.

## 2026-08-28 — Usage is measured; cost is configured

Input, cached-input, and output token counts plus the response model are stored beside successful results. Cost is calculated only when the operator supplies all three dated per-million-token rates. The code does not embed a permanent model price, and estimated cost is not treated as measured impact.
