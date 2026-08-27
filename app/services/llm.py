"""OpenAI Responses API client helpers."""

from __future__ import annotations

import json
import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DEFAULT_MODEL = "gpt-5.6-luna"


def get_model_name() -> str:
    return os.getenv("OPENAI_MODEL", DEFAULT_MODEL)


def is_api_configured() -> bool:
    return bool(os.getenv("OPENAI_API_KEY", "").strip())


def create_structured_response(
    *,
    instructions: str,
    user_input: str,
    schema_name: str,
    schema: dict[str, Any],
) -> dict[str, Any]:
    """Call the Responses API and return Structured Outputs JSON as a dict."""
    if not is_api_configured():
        raise RuntimeError(
            "OPENAI_API_KEY가 설정되어 있지 않습니다. "
            ".env.example을 참고해 환경변수를 설정하세요."
        )

    client = OpenAI()
    response = client.responses.create(
        model=get_model_name(),
        instructions=instructions,
        input=user_input,
        reasoning={"effort": "none"},
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

    if not response.output_text:
        raise RuntimeError("모델이 분류 결과를 반환하지 않았습니다.")

    return json.loads(response.output_text)
