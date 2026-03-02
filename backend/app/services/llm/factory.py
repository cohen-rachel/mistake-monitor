"""Factory for creating LLM provider instances based on config."""

from app.config import settings
from app.services.llm.base import LLMProvider

_cached_provider: LLMProvider | None = None


def get_llm_provider() -> LLMProvider:
    """Return the configured LLM provider instance (cached singleton)."""
    global _cached_provider
    if _cached_provider is not None:
        return _cached_provider

    provider = settings.llm_provider.lower()

    if provider == "ollama":
        from app.services.llm.local_adapter import OllamaProvider
        _cached_provider = OllamaProvider()

    elif provider == "openai":
        from app.services.llm.cloud_adapter import OpenAIProvider
        _cached_provider = OpenAIProvider()

    elif provider == "anthropic":
        from app.services.llm.cloud_adapter import AnthropicProvider
        _cached_provider = AnthropicProvider()

    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER: '{provider}'. "
            f"Valid options: ollama, openai, anthropic"
        )

    return _cached_provider
