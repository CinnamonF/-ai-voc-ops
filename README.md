# AI VOC Ops

AI-powered VOC intelligence and CX operations system for customer support analysis.

## Goal

Turn customer-support conversations into structured VOC signals that can be classified, evaluated, reviewed, and eventually used for trend / root-cause / CX operations workflows.

## Version roadmap

### v0.1 — VOC classifier

- fixed VOC taxonomy
- OpenAI Responses API + strict Structured Outputs
- priority / sentiment / human-review decision
- CSV batch analysis with row-level failure isolation
- measured token usage

### v0.2 — Evaluation & model improvement

- 200-row synthetic/provisional seed covering all 38 subcategories
- Major and Subcategory Accuracy / Macro Precision / Recall / F1
- High-risk Precision / Recall
- Human Review Precision / Recall / F1
- confusion matrices and per-subcategory metrics
- editable error analysis
- dataset / prompt / taxonomy / model provenance
- baseline-vs-candidate comparison

The v0.2 seed is **not a publishable gold benchmark yet**. Rows start as `label_status=provisional`; publishable metrics are calculated only from rows that a person has reviewed and promoted to `label_status=reviewed`.

### v0.3 — Pilot-ready web app

- dedicated **Pilot Test** page for one-VOC-at-a-time testing
- tester `Correct / Incorrect` feedback with corrected category, subcategory, priority, sentiment, and human-review label
- common email / phone / long-number patterns masked before the Pilot API call and before feedback persistence
- per-session single-test limit and configurable CSV batch limit
- session feedback CSV download
- optional persistent feedback through Supabase
- public app uses an INSERT-only anon policy; no anonymous feedback read access
- trusted local exporter converts persisted feedback into a `provisional` review queue
- tester feedback is never promoted to gold automatically

See `docs/v0.3_pilot.md` and `docs/supabase_pilot_feedback.sql`.

## Current UI

- Dashboard
- Pilot Test
- VOC Analyzer
- Taxonomy
- Evaluation Lab

## Local run

Python 3.11 or newer is recommended.

```bash
python -m venv .venv
```

Windows PowerShell:

```bash
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

macOS / Linux:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Set a local API key in `.env`:

```text
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-5.6-luna
OPENAI_REASONING_EFFORT=none
```

Run:

```bash
streamlit run app/main.py
```

## Tests and evaluation

Unit tests do not require a live API key:

```bash
pytest
```

Eight-case live smoke test:

```bash
python -m evals.live_smoke_test
```

Reviewed-label evaluation:

```bash
python -m evals.run_gold_eval
```

Exploratory provisional evaluation only:

```bash
python -m evals.run_gold_eval --include-provisional
```

Compare two evaluation runs:

```bash
python -m evals.compare_runs evals/results/<baseline>_summary.json evals/results/<candidate>_summary.json
```

## Pilot feedback persistence

The Pilot Test works without a database, but feedback is then limited to the current Streamlit session and should be downloaded before the session ends.

For persistent feedback, configure a Supabase project using:

```text
docs/supabase_pilot_feedback.sql
```

Public app configuration:

```text
SUPABASE_URL=...
SUPABASE_ANON_KEY=...
SUPABASE_FEEDBACK_TABLE=pilot_feedback
```

Never put a service-role key in the public app.

On a trusted local/admin machine, export feedback for manual review with:

```bash
python -m evals.export_pilot_feedback
```

The generated `pilot_feedback_review_queue.csv` is always `label_status=provisional` until manually reviewed.

## Pilot safety defaults

```text
PILOT_MAX_BATCH_ROWS=100
PILOT_MAX_TEXT_CHARS=4000
PILOT_MAX_SINGLE_ANALYSES_PER_SESSION=20
```

These are pilot safeguards, not production-grade internet rate limiting. Use synthetic or de-identified VOC only; never upload confidential former-employer customer data.

## Data policy

This repository uses public, synthetic, or deliberately de-identified data only. No real former-employer customer records or proprietary customer information should be used.

## Cost notes

Token counts are measured from API responses. Cost is calculated only when dated operator-supplied token prices are configured. No estimated cost, accuracy, recall, ROI, or business impact is presented as a measured project result unless it has actually been observed.

## Status

v0.1 classifier hardening, v0.2 evaluation infrastructure, and v0.3 pilot feedback workflow are implemented on their feature branches. Live performance metrics still require a configured API key and human-reviewed labels.
