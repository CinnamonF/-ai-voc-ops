import json
from decimal import Decimal
from types import SimpleNamespace

import pytest
from openai import OpenAIError

from app.services import llm


SCHEMA = {
    "type": "object",
    "properties": {"category": {"type": "string"}},
    "required": ["category"],
    "additionalProperties": False,
}


class FakeResponses:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        if self.error:
            raise self.error
        return self.response


def fake_client(*, response=None, error=None):
    responses = FakeResponses(response=response, error=error)
    return SimpleNamespace(responses=responses), responses


def completed_response(payload):
    return SimpleNamespace(
        status="completed",
        output_text=json.dumps(payload, ensure_ascii=False),
        model="gpt-5.6-luna-2026-08-01",
        usage=SimpleNamespace(
            input_tokens=120,
            output_tokens=24,
            input_tokens_details=SimpleNamespace(cached_tokens=40),
        ),
    )


def test_missing_api_key_has_clear_error(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(llm.LLMConfigurationError, match="OPENAI_API_KEY"):
        llm.create_structured_response(
            instructions="classify",
            user_input="message",
            schema_name="test",
            schema=SCHEMA,
        )


def test_placeholder_api_key_is_not_configured(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "your_api_key_here")

    assert llm.is_api_configured() is False


def test_successful_response_preserves_contract_and_usage(monkeypatch):
    monkeypatch.delenv("OPENAI_INPUT_COST_PER_1M", raising=False)
    monkeypatch.delenv("OPENAI_CACHED_INPUT_COST_PER_1M", raising=False)
    monkeypatch.delenv("OPENAI_OUTPUT_COST_PER_1M", raising=False)
    client, responses = fake_client(response=completed_response({"category": "배송"}))

    result = llm.create_structured_response(
        instructions="classify",
        user_input="배송 문의",
        schema_name="test",
        schema=SCHEMA,
        client=client,
    )

    assert result.data == {"category": "배송"}
    assert result.usage.input_tokens == 120
    assert result.usage.cached_input_tokens == 40
    assert result.usage.output_tokens == 24
    assert result.usage.model == "gpt-5.6-luna-2026-08-01"
    assert responses.kwargs["store"] is False
    assert responses.kwargs["reasoning"] == {"effort": "none"}
    assert responses.kwargs["text"]["format"] == {
        "type": "json_schema",
        "name": "test",
        "strict": True,
        "schema": SCHEMA,
    }


def test_api_failure_is_mapped_to_safe_error():
    client, _ = fake_client(error=OpenAIError("provider details"))

    with pytest.raises(llm.LLMRequestError, match="OpenAI API 요청에 실패"):
        llm.create_structured_response(
            instructions="classify",
            user_input="message",
            schema_name="test",
            schema=SCHEMA,
            client=client,
        )


def test_non_object_output_is_rejected():
    client, _ = fake_client(response=completed_response(["not", "an", "object"]))

    with pytest.raises(llm.LLMResponseError, match="객체 형식"):
        llm.create_structured_response(
            instructions="classify",
            user_input="message",
            schema_name="test",
            schema=SCHEMA,
            client=client,
        )


def test_estimated_cost_uses_cached_rate():
    pricing = llm.TokenPricing(
        input_per_million=Decimal("1.00"),
        cached_input_per_million=Decimal("0.10"),
        output_per_million=Decimal("2.00"),
    )

    estimate = llm.estimate_cost_usd(
        input_tokens=1000,
        cached_input_tokens=400,
        output_tokens=100,
        pricing=pricing,
    )

    assert estimate == pytest.approx(0.00084)
