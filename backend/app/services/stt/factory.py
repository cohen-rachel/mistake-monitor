"""Factory for creating STT provider instances based on config."""

from app.config import settings
from app.services.stt.base import STTProvider

_cached_provider: STTProvider | None = None


def get_stt_provider() -> STTProvider:
    """Return the configured STT provider instance (cached singleton)."""
    global _cached_provider
    if _cached_provider is not None:
        return _cached_provider

    provider = settings.stt_provider.lower()

    if provider == "dummy":
        from app.services.stt.dummy import DummySTTProvider
        _cached_provider = DummySTTProvider()

    elif provider == "whisper_api":
        from app.services.stt.whisper_api import WhisperAPIProvider
        _cached_provider = WhisperAPIProvider()

    elif provider == "whisper_local":
        from app.services.stt.whisper_local import WhisperLocalProvider
        _cached_provider = WhisperLocalProvider()

    else:
        raise ValueError(
            f"Unknown STT_PROVIDER: '{provider}'. "
            f"Valid options: dummy, whisper_api, whisper_local"
        )

    return _cached_provider
