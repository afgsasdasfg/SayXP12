"""Any OpenAI-compatible chat API: AITUNNEL (api.aitunnel.ru/v1), OpenRouter, DeepSeek, ...

Key from the env variable named in config (default OPENAI_API_KEY, put it in .env).
"""
from __future__ import annotations

import os

import httpx

from .base import LLM, LLMError, parse_json_loose


class OpenAICompatLLM(LLM):
    name = "openai"

    def __init__(self, model: str, base_url: str = "https://api.aitunnel.ru/v1",
                 api_key_env: str = "OPENAI_API_KEY", timeout: float = 10,
                 reasoning_effort: str | None = None, temperature: float | None = None,
                 json_mode: bool = True, max_tokens: int = 400, api_key: str | None = None):
        self.model = model
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.api_key = api_key or os.environ.get(api_key_env, "")
        if not self.api_key:
            raise LLMError(f"{api_key_env} is not set (put it in .env)")
        self.timeout = timeout
        self.reasoning_effort = reasoning_effort
        self.temperature = temperature
        self.json_mode = json_mode
        self.max_tokens = max_tokens
        self.last_usage: dict = {}

    async def complete_json(self, system: str, user: str) -> dict:
        body: dict = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_completion_tokens": self.max_tokens,
        }
        if self.json_mode:
            body["response_format"] = {"type": "json_object"}
        # reasoning models (gpt-5*): "minimal"/"low" keeps latency and hidden output tokens down
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        if self.temperature is not None:  # gpt-5 family rejects anything but the default
            body["temperature"] = self.temperature
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as c:
                r = await c.post(self.url, json=body, headers={"Authorization": f"Bearer {self.api_key}"})
        except httpx.HTTPError as e:
            raise LLMError(f"{self.url} unreachable: {e}") from e
        if r.status_code != 200:
            raise LLMError(f"HTTP {r.status_code}: {r.text[:300]}")
        data = r.json()
        self.last_usage = data.get("usage") or {}
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError) as e:
            raise LLMError(f"unexpected reply: {r.text[:300]}") from e
        return parse_json_loose(text)
