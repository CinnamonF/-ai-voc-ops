# OpenAI API Cost Notes

Last reviewed: 2026-08-28

The v0.1 classifier defaults to `gpt-5.6-luna`, a cost-sensitive GPT-5.6 model that supports the Responses API and Structured Outputs. The table below is a dated planning snapshot from the official model pages, not a permanent value in application code.

## Published token pricing

| Model | Input / 1M tokens | Cached input / 1M | Output / 1M tokens |
|---|---:|---:|---:|
| GPT-5.6 Luna | $0.20 | $0.02 | $1.20 |
| GPT-5.6 Terra | $2.00 | $0.20 | $12.00 |
| GPT-5.6 Sol | $4.00 | $0.40 | $20.00 |

Pricing can change. Re-check the official OpenAI pricing page before publishing portfolio cost claims.

Official sources reviewed on the date above:

- <https://developers.openai.com/api/docs/models/gpt-5.6-luna>
- <https://developers.openai.com/api/docs/models/gpt-5.6-terra>
- <https://developers.openai.com/api/docs/models/gpt-5.6-sol>

## Conservative v0.1 estimate

For planning, assume each ticket consumes roughly:

- 2,000 input tokens: taxonomy, boundary rules, JSON schema, and customer text
- 120 output tokens: category, subcategory, priority, sentiment, human-review flag, and reason

With GPT-5.6 Luna and no prompt-cache discount:

`cost per ticket ≈ (2,000 × $0.20 / 1M) + (120 × $1.20 / 1M) = $0.000544`

| Tickets | Estimated cost |
|---:|---:|
| 100 | $0.0544 |
| 1,000 | $0.544 |
| 5,000 | $2.72 |
| 10,000 | $5.44 |

This is an estimate, not a measured project result. Prompt caching may reduce repeated static-input cost.

## Runtime usage fields

Each successful analyzed row records:

- `input_tokens`
- `cached_input_tokens`
- `output_tokens`
- `model`
- `estimated_cost_usd`

The first four values come from the actual API response. `estimated_cost_usd` remains empty unless the operator configures all three dated rates:

```text
OPENAI_INPUT_COST_PER_1M=...
OPENAI_CACHED_INPUT_COST_PER_1M=...
OPENAI_OUTPUT_COST_PER_1M=...
```

This distinction prevents a token-price estimate from being mislabeled as an API-reported charge.

## Cost measurement plan

Before using a cost number in the portfolio:

1. Run the labeled evaluation set.
2. Export actual `input_tokens`, `cached_input_tokens`, and `output_tokens` from API responses.
3. Calculate cost from the exact model pricing at that date.
4. Publish measured cost per 1,000 VOCs alongside classification quality metrics.
