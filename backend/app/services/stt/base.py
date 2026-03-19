"""Abstract base class for STT providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class TranscriptSegment:
    text: str
    start: float = 0.0
    end: float = 0.0
    confidence: float = 1.0


@dataclass
class TranscriptResult:
    text: str
    segments: list[TranscriptSegment] = field(default_factory=list)
    average_confidence: float = 1.0


class STTProvider(ABC):
    """Abstract speech-to-text provider."""

    @abstractmethod
    async def transcribe(
        self,
        audio_bytes: bytes,
        language: str | None = None,
        filename: str | None = None,
        content_type: str | None = None,
    ) -> TranscriptResult:
        """Transcribe audio bytes and return a TranscriptResult."""
        ...

    @abstractmethod
    async def transcribe_chunk(
        self,
        audio_bytes: bytes,
        language: str | None = None,
        filename: str | None = None,
        content_type: str | None = None,
    ) -> TranscriptResult:
        """Transcribe a short audio chunk (for streaming)."""
        ...
