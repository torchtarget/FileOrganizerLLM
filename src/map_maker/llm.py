from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional

import httpx


@dataclass
class LLMResponse:
    content: str
    model: Optional[str] = None


class BaseLLM:
    def generate(self, system_prompt: str, user_prompt: str) -> LLMResponse:  # pragma: no cover - interface
        raise NotImplementedError


class FireworksLLM(BaseLLM):
    def __init__(self, model: str = "accounts/fireworks/models/llama-v3-70b-instruct"):
        self.api_key = os.getenv("FIREWORKS_API_KEY")
        self.model = os.getenv("FIREWORKS_MODEL", model)

    def generate(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        if not self.api_key:
            raise RuntimeError("FIREWORKS_API_KEY not configured")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload: Dict[str, object] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
        }
        response = httpx.post(
            "https://api.fireworks.ai/inference/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30.0,
        )
        response.raise_for_status()
        data = response.json()
        message = data["choices"][0]["message"]["content"]
        return LLMResponse(content=message, model=self.model)


class OllamaLLM(BaseLLM):
    def __init__(self, model: str = "llama3"):
        self.host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.model = os.getenv("OLLAMA_MODEL", model)

    def generate(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
        }
        response = httpx.post(
            f"{self.host}/api/chat",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=30.0,
        )
        response.raise_for_status()
        data = response.json()
        message = data.get("message", {}).get("content", "")
        return LLMResponse(content=message, model=self.model)


class StubLLM(BaseLLM):
    def generate(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        lines = [line.strip() for line in user_prompt.splitlines() if line.strip()]
        summary = " ".join(lines)
        truncated = summary[:600]
        return LLMResponse(content=truncated, model="stub")


def build_llm(provider: str) -> BaseLLM:
    if provider == "fireworks":
        return FireworksLLM()
    if provider == "ollama":
        return OllamaLLM()
    return StubLLM()


def safe_generate(llm: BaseLLM, system_prompt: str, user_prompt: str) -> LLMResponse:
    try:
        return llm.generate(system_prompt, user_prompt)
    except Exception as exc:  # pylint: disable=broad-except
        fallback = StubLLM()
        content = (
            "[fallback used due to error: "
            + str(exc)
            + "] "
            + fallback.generate(system_prompt, user_prompt).content
        )
        return LLMResponse(content=content, model="fallback")
