"""OpenAI transcription adapter for final STT."""

import logging
import httpx
from app.config import settings
from app.services.stt.base import STTProvider, TranscriptResult, TranscriptSegment
from app.services.stt.audio_utils import infer_audio_filename

logger = logging.getLogger(__name__)


class WhisperAPIProvider(STTProvider):
    """Uses OpenAI's transcription API for final speech-to-text."""

    _MODEL_NAME = "gpt-4o-mini-transcribe"

    def __init__(self):
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for whisper_api STT provider")
        self.api_key = settings.openai_api_key
        self.base_url = "https://api.openai.com/v1/audio/transcriptions"

    async def transcribe(
        self,
        audio_bytes: bytes,
        language: str | None = None,
        filename: str | None = None,
        content_type: str | None = None,
    ) -> TranscriptResult:
        return await self._call_whisper(audio_bytes, language, filename, content_type)

    async def transcribe_chunk(
        self,
        audio_bytes: bytes,
        language: str | None = None,
        filename: str | None = None,
        content_type: str | None = None,
    ) -> TranscriptResult:
        return await self._call_whisper(audio_bytes, language, filename, content_type)

    async def _call_whisper(
        self,
        audio_bytes: bytes,
        language: str | None,
        filename: str | None = None,
        content_type: str | None = None,
    ) -> TranscriptResult:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        upload_name = infer_audio_filename(
            filename=filename,
            content_type=content_type,
            default_stem="audio",
        )
        files = {
            "file": (
                upload_name,
                audio_bytes,
                content_type or "application/octet-stream",
            ),
        }
        data = {
            "model": self._MODEL_NAME,
            "response_format": "json",
        }
        if language:
            data["language"] = language

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(self.base_url, headers=headers, files=files, data=data)
            resp.raise_for_status()
            result = resp.json()

        text = result.get("text", "")
        segments = []
        for seg in result.get("segments", []):
            segments.append(
                TranscriptSegment(
                    text=seg.get("text", ""),
                    start=seg.get("start", 0.0),
                    end=seg.get("end", 0.0),
                    confidence=seg.get("avg_logprob", 0.0) + 1.0,  # normalize rough proxy
                )
            )

        avg_conf = sum(s.confidence for s in segments) / len(segments) if segments else 1.0
        logger.info(
            'Final STT Output (Model: %s, Language Hint: %s): "%s"',
            self._MODEL_NAME,
            language,
            text,
        )
        return TranscriptResult(text=text, segments=segments, average_confidence=avg_conf)
