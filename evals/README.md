# Evaluation

The classifier is evaluated against manually labeled synthetic or public examples. Evaluation numbers are reported only after the corresponding run has happened.

## Unit and contract tests

Run without an API key:

```bash
pytest
```

These tests mock the OpenAI boundary and cover the schema contract, taxonomy validation, human-review policy, usage extraction, safe failures, batch isolation, and smoke-runner reporting.

## Live smoke test

`smoke_cases.csv` contains eight boundary-focused cases. After setting a real local `OPENAI_API_KEY`, run:

```bash
python -m evals.live_smoke_test
```

For every case the command prints expected and predicted category paths, PASS/FAIL, priority, human review, and reason. It then prints major-category and subcategory accuracy. One request failure is recorded as a failed case without stopping later cases. The command exits nonzero if configuration is missing, a request fails, or any classification misses its expected path.

This is a billable live API check, not a statistically meaningful quality claim.

## Next evaluation stage

Expand `evaluation_dataset.csv` into a versioned gold set with enough examples per subcategory, then report:

- Major-category and subcategory accuracy
- Per-category recall and confusion pairs
- Critical/high-risk VOC recall
- Human-review precision and recall
- Token use and dated estimated cost per 1,000 VOCs
