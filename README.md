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
- Output is constrained by JSON Schema
- Category/subcategory pairs are validated again in application code
- API requests use `store=False`
- Failed rows are isolated in `analysis_error` instead of stopping an entire CSV run

## Run locally

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set your API key:

```text
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-5.6-luna
```

Never commit a real API key.

Then run:

```bash
streamlit run app/main.py
```

Run unit tests with:

```bash
pytest
```

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

See `docs/api_costs.md` for pricing assumptions and the measurement plan. Cost figures in the portfolio will be based on measured API usage rather than estimates.

## Status

🚧 Work in progress — v0.1 AI classifier and Streamlit analysis flow implemented. Live labeled evaluation is next.
