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
2. Classify each inquiry
3. Return category / subcategory / priority / sentiment
4. Flag cases requiring human review
5. Export classified results

## Current UI

The Streamlit prototype now includes:

- Dashboard
- VOC Analyzer
- Taxonomy
- Evaluation

Run locally with:

```bash
pip install -r requirements.txt
streamlit run app/main.py
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

## Status

🚧 Work in progress — v0.1 UI and taxonomy implemented; AI classifier implementation is next.
