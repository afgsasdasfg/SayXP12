"""Swappable LLM backend interface. Backends only turn text into a JSON object."""
from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod


class LLMError(RuntimeError):
    pass


class LLM(ABC):
    name = "base"

    @abstractmethod
    async def complete_json(self, system: str, user: str) -> dict:
        """Return the model's answer parsed as a JSON object (raise LLMError otherwise)."""


def parse_json_loose(text: str) -> dict:
    """Parse a JSON object, tolerating ```json fences or chatter around it."""
    text = text.strip()
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise LLMError(f"no JSON object in reply: {text[:200]!r}")
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        raise LLMError(f"invalid JSON: {e}") from e
    if not isinstance(obj, dict):
        raise LLMError("JSON is not an object")
    return obj


def make_llm(cfg: dict) -> LLM | None:
    """cfg = [llm] section. provider: gemini | ollama | rules (no LLM)."""
    provider = (cfg.get("provider") or "rules").lower()
    if provider == "gemini":
        from .gemini import GeminiLLM
        return GeminiLLM(model=cfg.get("gemini_model", "gemini-2.5-flash"),
                         timeout=cfg.get("timeout_s", 10))
    if provider == "ollama":
        from .ollama import OllamaLLM
        return OllamaLLM(model=cfg.get("ollama_model", "qwen2.5:7b"),
                         host=cfg.get("ollama_host", "http://localhost:11434"),
                         timeout=cfg.get("timeout_s", 20))
    return None
