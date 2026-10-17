"""Google Gemini (free tier) over REST. Key from env GEMINI_API_KEY (.env).

Free-tier limits change and prompts may be used for training; keep the backend swappable.
"""
from __future__ import annotations

import os

import httpx

from .base import LLM, LLMError, parse_json_loose

URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


class GeminiLLM(LLM):
    name = "gemini"

    def __init__(self, model: str = "gemini-2.5-flash", timeout: float = 10, api_key: str | None = None):
        self.model = model
        self.timeout = timeout
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not self.api_key:
            raise LLMError("GEMINI_API_KEY is not set (put it in .env)")

    async def complete_json(self, system: str, user: str) -> dict:
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"temperature": 0, "responseMimeType": "application/json"},
        }
        async with httpx.AsyncClient(timeout=self.timeout) as c:
            r = await c.post(URL.format(model=self.model), json=body,
                             headers={"x-goog-api-key": self.api_key})
        if r.status_code != 200:
            raise LLMError(f"gemini HTTP {r.status_code}: {r.text[:300]}")
        try:
            text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError) as e:
            raise LLMError(f"unexpected gemini reply: {r.text[:300]}") from e
        return parse_json_loose(text)
