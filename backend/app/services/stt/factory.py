"""Factory for creating STT provider instances based on config."""

from app.config import settings
from app.services.stt.base import STTProvider

_cached_providers: dict[str, STTProvider] = {}


def _build_provider(provider: str) -> STTProvider:
    provider = provider.lower()

    if provider == "dummy":
        from app.services.stt.dummy import DummySTTProvider
        return DummySTTProvider()

    if provider == "whisper_api":
        from app.services.stt.whisper_api import WhisperAPIProvider
        return WhisperAPIProvider()

    if provider == "whisper_local":
        from app.services.stt.whisper_local import WhisperLocalProvider
        return WhisperLocalProvider()

    raise ValueError(
        f"Unknown STT provider: '{provider}'. "
        f"Valid options: dummy, whisper_api, whisper_local"
    )


def _get_provider(provider: str) -> STTProvider:
    cached = _cached_providers.get(provider)
    if cached is not None:
        return cached
    built = _build_provider(provider)
    _cached_providers[provider] = built
    return built


def get_live_stt_provider() -> STTProvider:
    """Return the configured live STT provider."""
    provider = settings.live_stt_provider or settings.stt_provider
    return _get_provider(provider)


def get_final_stt_provider() -> STTProvider:
    """Return the configured final STT provider."""
    provider = settings.final_stt_provider or settings.stt_provider
    return _get_provider(provider)


def get_stt_provider() -> STTProvider:
    """Backwards-compatible alias for the final STT provider."""
    return get_final_stt_provider()
