"""OpenAI Responses API client and usage metadata helpers."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

load_dotenv()

DEFAULT_MODEL = "gpt-5.6-luna"
DEFAULT_REASONING_EFFORT = "none"
MAX_OUTPUT_TOKENS = 500
REASONING_EFFORTS = {"none", "low", "medium", "high", "xhigh", "max"}
PLACEHOLDER_API_KEYS = {"your_api_key_here", "sk-your-key-here"}


class LLMConfigurationError(RuntimeError):
    """Raised when local OpenAI configuration is missing or invalid."""


class LLMRequestError(RuntimeError):
    """Raised when the OpenAI request fails before a usable response arrives."""


class LLMResponseError(RuntimeError):
    """Raised when a completed response cannot satisfy the application contract."""


@dataclass(frozen=True)
class TokenPricing:
    """Operator-supplied token prices in USD per one million tokens."""

    input_per_million: Decimal
    cached_input_per_million: Decimal
    output_per_million: Decimal


@dataclass(frozen=True)
class UsageMetadata:
    """Measured API usage plus an optional estimate using dated operator inputs."""

    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    model: str | None
    estimated_cost_usd: float | None

    @classmethod
    def no_api_call(cls) -> "UsageMetadata":
        return cls(
            input_tokens=0,
            cached_input_tokens=0,
            output_tokens=0,
            model=None,
            estimated_cost_usd=None,
        )

    def as_columns(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "output_tokens": self.output_tokens,
            "model": self.model,
            "estimated_cost_usd": self.estimated_cost_usd,
        }


@dataclass(frozen=True)
class StructuredResponse:
    data: dict[str, Any]
    usage: UsageMetadata


def get_model_name() -> str:
    return os.getenv("OPENAI_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL


def get_reasoning_effort() -> str:
    effort = os.getenv("OPENAI_REASONING_EFFORT", DEFAULT_REASONING_EFFORT).strip()
    if effort not in REASONING_EFFORTS:
        allowed = ", ".join(sorted(REASONING_EFFORTS))
        raise LLMConfigurationError(
            f"OPENAI_REASONING_EFFORT 값이 올바르지 않습니다. 사용 가능: {allowed}"
        )
    return effort


def is_api_configured() -> bool:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    return bool(api_key and api_key not in PLACEHOLDER_API_KEYS)


def _get_api_key() -> str:
    if not is_api_configured():
        raise LLMConfigurationError(
            "OPENAI_API_KEY가 설정되어 있지 않습니다. "
            ".env.example을 참고해 환경변수를 설정하세요."
        )
    return os.environ["OPENAI_API_KEY"].strip()


def get_token_pricing() -> TokenPricing | None:
    """Read optional dated prices without embedding permanent prices in code."""
    names = {
        "input_per_million": "OPENAI_INPUT_COST_PER_1M",
        "cached_input_per_million": "OPENAI_CACHED_INPUT_COST_PER_1M",
        "output_per_million": "OPENAI_OUTPUT_COST_PER_1M",
    }
    raw_values = {field: os.getenv(env_name, "").strip() for field, env_name in names.items()}
    configured = [field for field, value in raw_values.items() if value]
    if not configured:
        return None
    if len(configured) != len(raw_values):
        raise LLMConfigurationError(
            "비용 추정을 사용하려면 OPENAI_*_COST_PER_1M 세 값을 모두 설정하세요."
        )

    try:
        values = {field: Decimal(value) for field, value in raw_values.items()}
    except InvalidOperation as exc:
        raise LLMConfigurationError("토큰 가격은 0 이상의 숫자로 설정하세요.") from exc
    if any(value < 0 for value in values.values()):
        raise LLMConfigurationError("토큰 가격은 0 이상의 숫자로 설정하세요.")
    return TokenPricing(**values)


def estimate_cost_usd(
    *,
    input_tokens: int,
    cached_input_tokens: int,
    output_tokens: int,
    pricing: TokenPricing | None,
) -> float | None:
    if pricing is None:
        return None
    uncached_input_tokens = max(input_tokens - cached_input_tokens, 0)
    million = Decimal(1_000_000)
    cost = (
        Decimal(uncached_input_tokens) * pricing.input_per_million
        + Decimal(cached_input_tokens) * pricing.cached_input_per_million
        + Decimal(output_tokens) * pricing.output_per_million
    ) / million
    return float(cost)


def _usage_from_response(response: Any, pricing: TokenPricing | None) -> UsageMetadata:
    usage = getattr(response, "usage", None)
    input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
    input_details = getattr(usage, "input_tokens_details", None)
    cached_input_tokens = int(getattr(input_details, "cached_tokens", 0) or 0)
    return UsageMetadata(
        input_tokens=input_tokens,
        cached_input_tokens=cached_input_tokens,
        output_tokens=output_tokens,
        model=getattr(response, "model", None),
        estimated_cost_usd=estimate_cost_usd(
            input_tokens=input_tokens,
            cached_input_tokens=cached_input_tokens,
            output_tokens=output_tokens,
            pricing=pricing,
        ),
    )


def create_structured_response(
    *,
    instructions: str,
    user_input: str,
    schema_name: str,
    schema: dict[str, Any],
    client: Any | None = None,
) -> StructuredResponse:
    """Call Responses with strict JSON Schema and return parsed data plus usage."""
    pricing = get_token_pricing()
    request_client = client or OpenAI(api_key=_get_api_key())
    try:
        response = request_client.responses.create(
            model=get_model_name(),
            instructions=instructions,
            input=user_input,
            reasoning={"effort": get_reasoning_effort()},
            max_output_tokens=MAX_OUTPUT_TOKENS,
            text={
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "strict": True,
                    "schema": schema,
                },
                "verbosity": "low",
            },
            store=False,
        )
    except OpenAIError as exc:
        raise LLMRequestError(
            "OpenAI API 요청에 실패했습니다. 연결 상태와 모델 접근 권한을 확인하세요."
        ) from exc

    if getattr(response, "status", None) != "completed":
        raise LLMResponseError("모델 분류가 완료되지 않았습니다. 잠시 후 다시 시도하세요.")

    output_text = str(getattr(response, "output_text", "") or "").strip()
    if not output_text:
        raise LLMResponseError("모델이 분류 결과를 반환하지 않았습니다.")

    try:
        data = json.loads(output_text)
    except json.JSONDecodeError as exc:
        raise LLMResponseError("모델 응답을 구조화된 결과로 읽지 못했습니다.") from exc
    if not isinstance(data, dict):
        raise LLMResponseError("모델 응답이 객체 형식이 아닙니다.")

    return StructuredResponse(data=data, usage=_usage_from_response(response, pricing))
