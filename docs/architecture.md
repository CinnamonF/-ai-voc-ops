# Architecture

## v0.1 execution flow

```text
Streamlit CSV upload
        |
        v
app/services/batch.py
  - preserve source row
  - isolate one failure per row
        |
        v
app/services/classifier.py
  - prompt and strict output schema derived from canonical taxonomy
  - application validation of fields and category/subcategory pair
  - deterministic human-review override for high-risk labels
        |
        v
app/services/llm.py
  - OpenAI Responses API
  - store=False
  - response.output_text parsing
  - response usage extraction
        |
        v
Structured result DataFrame
  - classification and reason
  - success/failure status
  - token usage, model, optional cost estimate
        |
        +--> VOC Analyzer filters and CSV download
        +--> Dashboard counts and distributions
```

## Module boundaries

- `app/services/taxonomy.py` is the only code-level taxonomy source. The schema, prompt labels, and Taxonomy page derive from it.
- `app/services/llm.py` owns SDK configuration, request construction, safe provider-error mapping, response parsing, and usage metadata.
- `app/services/classifier.py` owns the classification contract and business validation. Model output is never trusted solely because it passed JSON Schema.
- `app/services/batch.py` owns row preservation, failure isolation, and operational counts. Streamlit pages do not implement classification loops.
- `app/pages/*` only handles user interaction and presentation.

## Trust and data boundaries

Uploaded CSV text is untrusted input and is sent as user input, separate from classifier instructions. The model has no tools, the response is schema-constrained, and application code rejects invalid labels or label pairs. Unexpected exception details are not written to downloadable results.

The API key is read from the local environment and `.env` is ignored by Git. v0.1 is a local prototype for synthetic or public examples; production authentication, access control, PII redaction, rate limiting, retry/backoff, and CRM ingestion remain out of scope.

## Future modules

`root_cause.py` and `trend_detector.py` remain placeholders. They should consume validated result rows only after the classifier has a larger labeled evaluation set.
