"""Helpers for preserving audio container hints across STT providers."""

from pathlib import Path


_CONTENT_TYPE_TO_SUFFIX = {
    "audio/m4a": ".m4a",
    "audio/mp4": ".m4a",
    "audio/x-m4a": ".m4a",
    "audio/aac": ".aac",
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/wave": ".wav",
    "audio/webm": ".webm",
    "audio/ogg": ".ogg",
    "audio/opus": ".opus",
    "audio/flac": ".flac",
}


def infer_audio_suffix(
    filename: str | None = None,
    content_type: str | None = None,
    default: str = ".webm",
) -> str:
    if filename:
        suffix = Path(filename).suffix.lower()
        if suffix:
            return suffix
    if content_type:
        normalized = content_type.split(";", 1)[0].strip().lower()
        return _CONTENT_TYPE_TO_SUFFIX.get(normalized, default)
    return default


def infer_audio_filename(
    filename: str | None = None,
    content_type: str | None = None,
    default_stem: str = "audio",
) -> str:
    suffix = infer_audio_suffix(filename=filename, content_type=content_type)
    if filename:
        return filename
    return f"{default_stem}{suffix}"
