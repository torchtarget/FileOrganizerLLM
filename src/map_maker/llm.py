from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import httpx
from pydantic import BaseModel

try:
    from fireworks import LLM as FireworksLLM_SDK
    FIREWORKS_AVAILABLE = True
except ImportError:
    FIREWORKS_AVAILABLE = False


@dataclass
class LLMResponse:
    content: str
    model: Optional[str] = None


class BaseLLM:
    def generate(self, system_prompt: str, user_prompt: str, response_schema: Optional[type[BaseModel]] = None) -> LLMResponse:  # pragma: no cover - interface
        raise NotImplementedError


class FireworksLLM(BaseLLM):
    def __init__(self, model: str = "llama4-maverick-instruct-basic"):
        if not FIREWORKS_AVAILABLE:
            raise RuntimeError("fireworks-ai package not installed. Run: pip install fireworks-ai")

        # Get API key from environment
        self.api_key = os.getenv("FIREWORKS_API_KEY")
        if not self.api_key:
            raise RuntimeError("FIREWORKS_API_KEY environment variable not set")

        model_name = os.getenv("FIREWORKS_MODEL", model)

        # Initialize LLM with deployment_type="auto" and explicit api_key
        self.client = FireworksLLM_SDK(
            model=model_name,
            deployment_type="auto",
            api_key=self.api_key
        )
        self.model = self.client.model

        # Embedding model
        self.embedding_model = "nomic-ai/nomic-embed-text-v1.5"

    def generate(self, system_prompt: str, user_prompt: str, response_schema: Optional[type[BaseModel]] = None) -> LLMResponse:
        try:
            kwargs = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.2,
            }

            # Add structured output if schema provided
            if response_schema:
                kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": response_schema.__name__,
                        "schema": response_schema.model_json_schema()
                    }
                }

            response = self.client.chat.completions.create(**kwargs)
            message = response.choices[0].message.content
            return LLMResponse(content=message, model=self.model)
        except Exception as e:
            raise RuntimeError(f"Fireworks API error: {e}")

    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding vector for semantic search"""
        try:
            response = httpx.post(
                "https://api.fireworks.ai/inference/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.embedding_model,
                    "input": text,
                },
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()
            return data["data"][0]["embedding"]
        except Exception as e:
            print(f"Warning: Failed to generate embedding: {e}")
            return None


class OllamaLLM(BaseLLM):
    def __init__(self, model: str = "llama3"):
        self.host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.model = os.getenv("OLLAMA_MODEL", model)

    def generate(self, system_prompt: str, user_prompt: str, response_schema: Optional[type[BaseModel]] = None) -> LLMResponse:
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
    def generate(self, system_prompt: str, user_prompt: str, response_schema: Optional[type[BaseModel]] = None) -> LLMResponse:
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


def safe_generate(llm: BaseLLM, system_prompt: str, user_prompt: str, response_schema: Optional[type[BaseModel]] = None) -> LLMResponse:
    try:
        return llm.generate(system_prompt, user_prompt, response_schema=response_schema)
    except Exception as exc:  # pylint: disable=broad-except
        fallback = StubLLM()
        content = (
            "[fallback used due to error: "
            + str(exc)
            + "] "
            + fallback.generate(system_prompt, user_prompt).content
        )
        return LLMResponse(content=content, model="fallback")
