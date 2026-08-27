# Prompt v0.2 Experiment

Status: **blocked pending reviewed labels + live baseline**

## Why this experiment is not pre-filled

A prompt improvement should respond to observed failure modes. Creating a candidate prompt first and then claiming improvement would bias the experiment.

The current 200-row seed is synthetic and `provisional`, and the remote repository does not contain an API key. Therefore no baseline/candidate accuracy is claimed here.

## Baseline

- Classifier prompt: v0.1
- Taxonomy: v0.1
- Dataset: v0.2 seed, only rows with `label_status=reviewed`
- Model: exact response model recorded by the API

Run:

```bash
python -m evals.run_gold_eval
```

Record:

- Major Accuracy / Macro F1
- Subcategory Accuracy / Macro F1
- High-risk Recall
- Human Review Precision / Recall
- confusion matrices
- annotated error analysis
- measured input / cached input / output tokens

## Candidate hypothesis

Do **not** write the v0.2 prompt hypothesis until baseline error analysis identifies a repeated failure mode.

Examples of acceptable evidence-driven hypotheses:

- repeated `부분 환불` vs `환불 금액` confusion -> sharpen state-vs-calculation boundary
- repeated `배송완료 미수령` vs `배송 중 분실` confusion -> emphasize system delivery status
- repeated `상품 불량` vs `파손` confusion -> emphasize functional vs physical damage evidence

Document the selected hypothesis here before changing the prompt.

## Comparison rule

Use the exact same reviewed dataset rows for baseline and candidate.

```bash
python -m evals.compare_runs \
  evals/results/<baseline>_summary.json \
  evals/results/<candidate>_summary.json
```

A candidate should not be accepted only because overall accuracy rises. Check high-risk recall and human-review precision/recall for regressions.

## Result

Pending. No performance result has been measured or claimed.
