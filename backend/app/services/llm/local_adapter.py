"""Ollama (local LLM) adapter."""

import httpx
import logging
from app.config import settings
from app.services.llm.base import LLMProvider

logger = logging.getLogger(__name__)
class OllamaProvider(LLMProvider):
    """Calls a local Ollama instance for LLM completions."""

    def __init__(self):
        self.base_url = settings.ollama_base_url
        self.model = settings.ollama_model

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "format": "json",
        }
        logger.info(f"LLM Payload: {payload}")
        logger.debug("Sending LLM request to {url}")
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(url, json=payload)
            logger.debug(f"LLM Response: {resp.json()}")
            resp.raise_for_status()
            data = resp.json()
            logger.debug(f"LLM Data: {data}")

        return data.get("message", {}).get("content", "")
