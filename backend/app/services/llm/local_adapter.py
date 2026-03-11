"""Ollama (local LLM) adapter."""

import httpx
import logging
from typing import Any

from app.config import settings
from app.services.llm.base import LLMProvider

logger = logging.getLogger(__name__)


class OllamaProvider(LLMProvider):
    """Calls a local Ollama instance for LLM completions."""

    def __init__(self):
        self.base_url = settings.ollama_base_url
        self.model = settings.ollama_model
        self.timeout_seconds = settings.ollama_timeout_seconds

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        url = f"{self.base_url}/api/chat"
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "format": "json",
        }
        logger.info("LLM Payload: %s", payload)
        logger.debug(f"Sending LLM request to {url}")

        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            try:
                resp = await client.post(url, json=payload)
                resp_text = resp.text
                logger.debug("LLM Response text (truncated): %s", resp_text[:500])
                resp.raise_for_status()
                data = resp.json()
                logger.debug("LLM Data: %s", data)
            except httpx.HTTPStatusError as exc:
                body = exc.response.text if exc.response is not None else ""
                logger.exception(
                    "LLM responded with %s; payload keys=%s; response snippet=%s",
                    exc.response.status_code if exc.response else "unknown",
                    list(payload.keys()),
                    body[:500],
                )
                raise
            except httpx.HTTPError as exc:
                logger.exception(
                    "LLM request failed (%s); timeout=%ss; payload keys=%s",
                    type(exc).__name__,
                    self.timeout_seconds,
                    list(payload.keys()),
                )
                raise

        return data.get("message", {}).get("content", "")
