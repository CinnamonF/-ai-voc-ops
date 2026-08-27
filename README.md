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

## Current UI

The Streamlit prototype includes:

- Dashboard
- VOC Analyzer
- Taxonomy
- Evaluation

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

The live smoke test makes billable API requests. Its results are exploratory and are not treated as final evaluation metrics.

## Result columns

The analyzer preserves every input column and appends classification, operational, and usage fields:

- `category`, `subcategory`, `priority`, `sentiment`, `requires_human_review`, `reason`
- `analysis_status`, `analysis_error`
- `input_tokens`, `cached_input_tokens`, `output_tokens`, `model`
- `estimated_cost_usd` only when all three dated token-price environment variables are configured

## Project Structure

```text
app/        Application code
data/       Sample and processed datasets
docs/       CX/VOC design documents
evals/      Evaluation datasets and scripts
tests/      Automated tests
```

## Data Policy

This project uses public or synthetic customer-support data only.
No confidential customer data or former-employer proprietary data is included.

## Cost notes

See `docs/api_costs.md` for pricing assumptions and the measurement plan. Token counts are measured from each API response. A cost remains an estimate calculated from operator-supplied, dated prices and must not be presented as business performance.

## Status

v0.1 classifier hardening and the Streamlit analysis flow are implemented. The eight-case live smoke run still requires a locally configured API key; no live accuracy result is published in this repository.
