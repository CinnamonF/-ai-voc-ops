# AI VOC Ops

AI-powered VOC intelligence and CX operations system for customer support analysis.

## Goal

Transform raw customer support conversations into structured VOC insights that support:

- VOC classification
- Trend detection
- Root-cause analysis
- 상담 품질(QA) 평가
- Human handoff 판단
- Weekly CX reporting

## v0.1 MVP

1. Upload VOC CSV
2. Classify each inquiry with a fixed VOC taxonomy
3. Return category / subcategory / priority / sentiment
4. Flag cases requiring human review
5. Export classified results

## v0.2 Evaluation & Model Improvement

v0.2 adds a repeatable evaluation loop instead of adding more AI features.

- 200-row synthetic/provisional evaluation seed covering all 38 subcategories
- Major category Accuracy / Macro Precision / Recall / F1
- Subcategory Accuracy / Macro Precision / Recall / F1
- High-risk Precision / Recall
- Human Review Precision / Recall / F1
- Major and subcategory confusion matrices
- Per-subcategory metrics
- Editable error-analysis workflow
- Model / prompt / taxonomy / dataset provenance
- Aggregated measured token usage
- Baseline-vs-candidate run comparison

The 200-row seed is **not a publishable gold benchmark yet**. Every row starts with
`label_status=provisional`. Portfolio metrics are calculated by default only from
rows that a person has reviewed and changed to `label_status=reviewed`.

See `docs/v0.2_evaluation.md`.

## Current UI

The Streamlit prototype includes:

- Dashboard
- VOC Analyzer
- Taxonomy
- Evaluation Lab

## AI classifier

The v0.1 classifier uses the OpenAI Responses API with Structured Outputs.

- Default model: `gpt-5.6-luna`
- Output is constrained by a strict JSON Schema derived from the canonical taxonomy
- Category/subcategory pairs and every auxiliary label are validated again in application code
- High-risk cases are deterministically forced to human review after model output
- API requests use `store=False`
- Failed rows retain their original input and are isolated with `analysis_status` and `analysis_error`
- Successful rows retain measured token usage and the response model

## Run locally

Python 3.11 or newer is recommended.

### Windows PowerShell

```bash
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Edit the local `.env` file and replace the placeholder key:

```text
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-5.6-luna
OPENAI_REASONING_EFFORT=none
```

`.env` is ignored by Git. Never commit or paste a real API key into source files.

Start Streamlit from the repository root:

```bash
streamlit run app/main.py
```

Run unit tests without a live API call:

```bash
pytest
```

Run the hand-labeled live smoke cases after configuring an API key:

```bash
python -m evals.live_smoke_test
```

Run the v0.2 evaluation seed:

```bash
python -m evals.run_gold_eval
```

The default command saves predictions but will not calculate publishable metrics while
the seed labels remain provisional.

For development-only pipeline checks:

```bash
python -m evals.run_gold_eval --include-provisional
```

Any metrics produced with `--include-provisional` are exploratory and must not be
presented as portfolio performance.

Compare two evaluation summaries:

```bash
python -m evals.compare_runs evals/results/<baseline>_summary.json evals/results/<candidate>_summary.json
```

## Result columns

The analyzer preserves every input column and appends classification, operational, and usage fields:

- `category`, `subcategory`, `priority`, `sentiment`, `requires_human_review`, `reason`
- `analysis_status`, `analysis_error`
- `input_tokens`, `cached_input_tokens`, `output_tokens`, `model`
- `estimated_cost_usd` only when all three dated token-price environment variables are configured

Evaluation runs additionally preserve explicit dataset/prompt/taxonomy versions.

## Project Structure

```text
app/        Application code
data/       Sample and processed datasets
docs/       CX/VOC design documents
evals/      Versioned evaluation datasets, runners, and result artifacts
tests/      Automated tests
```

## Data Policy

This project uses public or synthetic customer-support data only.
No confidential customer data or former-employer proprietary data is included.

## Cost notes

See `docs/api_costs.md` for pricing assumptions and the measurement plan. Token counts are measured from each API response. A cost remains an estimate calculated from operator-supplied, dated prices and must not be presented as business performance.

## Status

v0.1 classifier hardening is implemented. v0.2 evaluation infrastructure and a 200-row
synthetic/provisional seed are implemented on the evaluation branch. Live model quality
metrics still require an API key and human-reviewed labels; no accuracy or ROI result is
claimed in this repository.
