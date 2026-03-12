"""Local Whisper (faster-whisper) adapter for STT.

Requires: pip install faster-whisper
This is optional — only installed if you want local STT.
"""

import logging
import tempfile
import os
from app.services.stt.base import STTProvider, TranscriptResult, TranscriptSegment
from app.services.stt.audio_utils import infer_audio_suffix

logger = logging.getLogger(__name__)


class WhisperLocalProvider(STTProvider):
    """Uses faster-whisper for local speech-to-text. Loads model on first use."""

    def __init__(self, model_size: str = "base"):
        self.model_size = model_size
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError:
                raise ImportError(
                    "faster-whisper is not installed. "
                    "Install with: pip install faster-whisper"
                )
            logger.info(f"Loading faster-whisper model '{self.model_size}' (first load downloads ~140MB)...")
            self._model = WhisperModel(self.model_size, compute_type="int8")
            logger.info("faster-whisper model loaded.")
        return self._model

    async def transcribe(
        self,
        audio_bytes: bytes,
        language: str = "en",
        filename: str | None = None,
        content_type: str | None = None,
    ) -> TranscriptResult:
        return await self._run_whisper(audio_bytes, language, filename, content_type)

    async def transcribe_chunk(
        self,
        audio_bytes: bytes,
        language: str = "en",
        filename: str | None = None,
        content_type: str | None = None,
    ) -> TranscriptResult:
        return await self._run_whisper(audio_bytes, language, filename, content_type)

    async def _run_whisper(
        self,
        audio_bytes: bytes,
        language: str,
        filename: str | None = None,
        content_type: str | None = None,
    ) -> TranscriptResult:
        import asyncio

        if not audio_bytes or len(audio_bytes) < 100:
            return TranscriptResult(text="", segments=[], average_confidence=0.0)

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._sync_transcribe,
            audio_bytes,
            language,
            filename,
            content_type,
        )

    def _sync_transcribe(
        self,
        audio_bytes: bytes,
        language: str,
        filename: str | None = None,
        content_type: str | None = None,
    ) -> TranscriptResult:
        model = self._get_model()

        # Write audio to temp file (faster-whisper needs a file path)
        suffix = infer_audio_suffix(filename=filename, content_type=content_type)
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(audio_bytes)
            tmp_path = f.name

        try:
            segments_iter, info = model.transcribe(
                tmp_path,
                language=language,
                vad_filter=True,  # filter out silence
                vad_parameters=dict(min_silence_duration_ms=300),
            )

            segments = []
            full_text_parts = []
            for seg in segments_iter:
                text = seg.text.strip()
                if not text:
                    continue
                # Clamp confidence to [0, 1] range
                raw_confidence = min(max(seg.avg_logprob + 1.0, 0.0), 1.0)
                segments.append(
                    TranscriptSegment(
                        text=text,
                        start=seg.start,
                        end=seg.end,
                        confidence=raw_confidence,
                    )
                )
                full_text_parts.append(text)

            text = " ".join(full_text_parts)
            avg_conf = sum(s.confidence for s in segments) / len(segments) if segments else 0.0
            logger.info(f"STT Output (Language: {language}): \"{text}\" (Avg Confidence: {avg_conf:.2f})") # ADD THIS LINE
            return TranscriptResult(text=text, segments=segments, average_confidence=avg_conf)
        except Exception as e:
            logger.error(f"faster-whisper transcription failed: {e}")
            return TranscriptResult(text="", segments=[], average_confidence=0.0)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
