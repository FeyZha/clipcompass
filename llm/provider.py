"""Single provider boundary for product LLM calls."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class LLMError(RuntimeError):
    pass


@dataclass(frozen=True)
class GenerationResult:
    value: Any
    model: str
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int
    latency_ms: float
    error: str | None = None

    def usage(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "cache_hit": self.cached_input_tokens > 0,
            "latency_ms": self.latency_ms,
            "error": self.error,
        }


def load_project_env() -> None:
    path = PROJECT_ROOT / ".env"
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        value = value.strip().strip('"').strip("'")
        if name:
            os.environ.setdefault(name, value)


def _provider():
    load_project_env()
    provider = os.getenv("LLM_PROVIDER", "deepseek").strip().lower()
    if provider != "deepseek":
        raise LLMError(f"Unsupported LLM_PROVIDER: {provider}. Only deepseek is allowed.")
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise LLMError("DEEPSEEK_API_KEY is not configured.")
    model = os.getenv("LLM_MODEL", "deepseek-flash").strip()
    if model != "deepseek-flash":
        raise LLMError(f"Unsupported LLM_MODEL: {model}. Product runtime requires deepseek-flash.")
    from .deepseek import DeepSeekProvider

    return DeepSeekProvider(api_key=api_key, model=model)


def generate_json(
    system_prompt: str,
    user_prompt: str,
    *,
    schema: dict[str, Any],
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> GenerationResult:
    schema_instruction = "\nReturn one JSON object matching this schema exactly:\n" + json.dumps(
        schema, ensure_ascii=False, separators=(",", ":")
    )
    result = _provider().generate(
        system_prompt + schema_instruction,
        user_prompt,
        json_output=True,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    try:
        value = json.loads(result.value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise LLMError("DeepSeek returned invalid JSON.") from exc
    if not isinstance(value, dict):
        raise LLMError("DeepSeek JSON response must be an object.")
    return GenerationResult(value=value, **{key: value for key, value in result.__dict__.items() if key != "value"})


def generate_text(
    system_prompt: str,
    user_prompt: str,
    *,
    temperature: float = 0.0,
    max_tokens: int = 1024,
) -> GenerationResult:
    return _provider().generate(
        system_prompt,
        user_prompt,
        json_output=False,
        temperature=temperature,
        max_tokens=max_tokens,
    )
