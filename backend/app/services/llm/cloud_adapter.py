"""Cloud LLM adapters (OpenAI / Anthropic)."""

import httpx
import logging
from app.config import settings
from app.services.llm.base import LLMProvider

logger = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
    """Calls the OpenAI Chat Completions API."""

    def __init__(self):
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for openai LLM provider")
        self.api_key = settings.openai_api_key
        self.model = settings.openai_model

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
        }
        if not self.model.startswith("gpt-5"):
            payload["temperature"] = 0.2

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(url, headers=headers, json=payload)
            if resp.is_error:
                logger.error(
                    "OpenAI chat completion failed with %s: %s",
                    resp.status_code,
                    resp.text[:1000],
                )
            resp.raise_for_status()
            data = resp.json()

        return data["choices"][0]["message"]["content"]


class AnthropicProvider(LLMProvider):
    """Calls the Anthropic Messages API."""

    def __init__(self):
        if not settings.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is required for anthropic LLM provider")
        self.api_key = settings.anthropic_api_key
        self.model = settings.anthropic_model

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "max_tokens": 4096,
            "system": system_prompt,
            "messages": [
                {"role": "user", "content": user_prompt},
            ],
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

        return data["content"][0]["text"]
