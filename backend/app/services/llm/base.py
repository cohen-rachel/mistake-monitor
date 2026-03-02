"""Abstract base class for LLM providers."""

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Abstract LLM provider for text completion."""

    @abstractmethod
    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Send system + user prompt and return raw string response (expected JSON)."""
        ...
