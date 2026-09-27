"""Project-wide LLM provider boundary."""

from .provider import GenerationResult, LLMError, generate_json, generate_text

__all__ = ["GenerationResult", "LLMError", "generate_json", "generate_text"]
