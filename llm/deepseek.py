"""DeepSeek official API implementation."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

from .provider import GenerationResult, LLMError


class DeepSeekProvider:
    endpoint = "https://api.deepseek.com/chat/completions"

    def __init__(self, *, api_key: str, model: str) -> None:
        self.api_key = api_key
        self.model = model

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        json_output: bool,
        temperature: float,
        max_tokens: int,
    ) -> GenerationResult:
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "thinking": {"type": "disabled"},
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if json_output:
            body["response_format"] = {"type": "json_object"}
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:1000]
            raise LLMError(f"DeepSeek API HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise LLMError(f"DeepSeek API request failed: {exc}") from exc
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            raise LLMError("DeepSeek API returned an invalid response.") from exc
        latency_ms = round((time.perf_counter() - started) * 1000, 3)
        try:
            content = payload["choices"][0]["message"]["content"]
            usage = payload.get("usage") or {}
            details = usage.get("prompt_tokens_details") or {}
            cached = details.get("cached_tokens", usage.get("prompt_cache_hit_tokens", 0)) or 0
            if not isinstance(content, str) or not content.strip():
                raise ValueError("empty content")
            return GenerationResult(
                value=content,
                model=str(payload.get("model") or self.model),
                input_tokens=int(usage.get("prompt_tokens") or 0),
                output_tokens=int(usage.get("completion_tokens") or 0),
                cached_input_tokens=int(cached),
                latency_ms=latency_ms,
            )
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise LLMError("DeepSeek API returned an incomplete response.") from exc
