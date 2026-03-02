"""Dummy STT provider for testing without ML dependencies."""

from app.services.stt.base import STTProvider, TranscriptResult, TranscriptSegment

DUMMY_TRANSCRIPT = (
    "Yesterday I go to the store and buyed some foods. "
    "I was looking for a good bread but they didn't had none. "
    "Then I goes to the park and seen my friend. "
    "He told me that he don't like the weathers today. "
    "We was talking about the movie what we seen last week."
)

DUMMY_SEGMENTS = [
    TranscriptSegment(text="Yesterday I go to the store and buyed some foods.", start=0.0, end=4.2, confidence=0.92),
    TranscriptSegment(text="I was looking for a good bread but they didn't had none.", start=4.3, end=8.1, confidence=0.88),
    TranscriptSegment(text="Then I goes to the park and seen my friend.", start=8.2, end=11.5, confidence=0.85),
    TranscriptSegment(text="He told me that he don't like the weathers today.", start=11.6, end=15.0, confidence=0.90),
    TranscriptSegment(text="We was talking about the movie what we seen last week.", start=15.1, end=19.0, confidence=0.87),
]

_chunk_index = 0


class DummySTTProvider(STTProvider):
    """Returns canned transcript for testing. No ML dependencies required."""

    async def transcribe(self, audio_bytes: bytes, language: str = "en") -> TranscriptResult:
        return TranscriptResult(
            text=DUMMY_TRANSCRIPT,
            segments=DUMMY_SEGMENTS,
            average_confidence=0.884,
        )

    async def transcribe_chunk(self, audio_bytes: bytes, language: str = "en") -> TranscriptResult:
        global _chunk_index
        idx = _chunk_index % len(DUMMY_SEGMENTS)
        _chunk_index += 1
        seg = DUMMY_SEGMENTS[idx]
        return TranscriptResult(
            text=seg.text,
            segments=[seg],
            average_confidence=seg.confidence,
        )
