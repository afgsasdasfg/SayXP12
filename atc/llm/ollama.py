"""Local model through Ollama (/api/chat with format=json)."""
from __future__ import annotations

import httpx

from .base import LLM, LLMError, parse_json_loose


class OllamaLLM(LLM):
    name = "ollama"

    def __init__(self, model: str = "qwen2.5:7b", host: str = "http://localhost:11434", timeout: float = 20):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout

    async def complete_json(self, system: str, user: str) -> dict:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "format": "json",
            "stream": False,
            "options": {"temperature": 0},
            "keep_alive": "30m",
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as c:
                r = await c.post(f"{self.host}/api/chat", json=body)
        except httpx.HTTPError as e:
            raise LLMError(f"ollama unreachable: {e}") from e
        if r.status_code != 200:
            raise LLMError(f"ollama HTTP {r.status_code}: {r.text[:300]}")
        return parse_json_loose(r.json()["message"]["content"])
