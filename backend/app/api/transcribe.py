"""Transcription endpoints: REST and WebSocket streaming."""

import base64
import json
import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, UploadFile, File, Form

from app.services.stt.factory import get_stt_provider

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/transcribe", tags=["transcribe"])


def _normalized_words(text: str) -> list[str]:
    return [word.lower().strip(".,!?;:\"'()[]{}") for word in text.split()]


def _shared_prefix_length(previous_text: str, current_text: str) -> int:
    previous_words = previous_text.split()
    current_words = current_text.split()
    previous_normalized = _normalized_words(previous_text)
    current_normalized = _normalized_words(current_text)

    shared_prefix = 0
    max_shared = min(len(previous_words), len(current_words))
    while shared_prefix < max_shared:
        if previous_normalized[shared_prefix] != current_normalized[shared_prefix]:
            break
        shared_prefix += 1

    return shared_prefix


@router.post("")
async def transcribe_audio(
    audio_file: UploadFile = File(...),
    language: str = Form("en"),
):
    """Transcribe an uploaded audio file and return the transcript."""
    audio_bytes = await audio_file.read()
    stt = get_stt_provider()
    result = await stt.transcribe(
        audio_bytes,
        language,
        filename=audio_file.filename,
        content_type=audio_file.content_type,
    )

    return {
        "text": result.text,
        "segments": [
            {
                "text": seg.text,
                "start": seg.start,
                "end": seg.end,
                "confidence": seg.confidence,
            }
            for seg in result.segments
        ],
        "average_confidence": result.average_confidence,
    }


def _decode_base64_audio(payload: str) -> bytes:
    try:
        return base64.b64decode(payload, validate=True)
    except Exception as exc:
        raise ValueError("Invalid base64 audio payload") from exc


@router.websocket("/stream")
async def transcribe_stream(websocket: WebSocket):
    """WebSocket endpoint for real-time audio transcription.

    Client sends binary audio chunks. Server responds with JSON transcription results.
    Send a text message 'stop' to end the session.
    """
    await websocket.accept()
    stt = get_stt_provider()
    language = websocket.query_params.get("language", "en")
    stream_filename = websocket.query_params.get("filename")
    stream_content_type = websocket.query_params.get("content_type")
    chunk_index = 0
    cumulative_audio = bytearray()
    previous_full_text = ""
    committed_word_count = 0

    try:
        while True:
            # Receive data — could be binary (audio) or text (control)
            data = await websocket.receive()

            if data.get("type") == "websocket.disconnect":
                break

            if "text" in data:
                text_msg = data["text"]
                if text_msg.strip().lower() == "stop":
                    if previous_full_text:
                        final_words = previous_full_text.split()
                        remaining_text = " ".join(final_words[committed_word_count:]).strip()
                        if remaining_text:
                            logger.info(
                                "WS transcript payload chunk=%s incremental=%r full=%r",
                                chunk_index,
                                remaining_text,
                                previous_full_text,
                            )
                            await websocket.send_json({
                                "type": "transcript",
                                "text": remaining_text,
                                "full_text": previous_full_text,
                                "chunk_index": chunk_index,
                                "is_final": True,
                            })
                            committed_word_count = len(final_words)
                    await websocket.send_json({
                        "type": "stopped",
                        "chunk_index": chunk_index,
                    })
                    break
                try:
                    message = json.loads(text_msg)
                except json.JSONDecodeError:
                    continue

                msg_type = message.get("type")
                if msg_type == "config":
                    language = message.get("language", language) or language
                    stream_filename = message.get("filename") or message.get("file_name") or stream_filename
                    stream_content_type = message.get("content_type") or stream_content_type
                    continue

                if msg_type == "stop":
                    if previous_full_text:
                        final_words = previous_full_text.split()
                        remaining_text = " ".join(final_words[committed_word_count:]).strip()
                        if remaining_text:
                            await websocket.send_json({
                                "type": "transcript",
                                "text": remaining_text,
                                "full_text": previous_full_text,
                                "chunk_index": chunk_index,
                                "is_final": True,
                            })
                    await websocket.send_json({
                        "type": "stopped",
                        "chunk_index": chunk_index,
                    })
                    break

                if msg_type != "audio_chunk":
                    continue

                payload = message.get("data")
                if not payload:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Missing audio chunk payload",
                        "chunk_index": chunk_index,
                    })
                    continue
                try:
                    audio_bytes = _decode_base64_audio(payload)
                except ValueError as exc:
                    await websocket.send_json({
                        "type": "error",
                        "message": str(exc),
                        "chunk_index": chunk_index,
                    })
                    continue

                stream_filename = message.get("filename") or message.get("file_name") or stream_filename
                stream_content_type = message.get("content_type") or stream_content_type

                try:
                    cumulative_audio.extend(audio_bytes)
                    result = await stt.transcribe(
                        bytes(cumulative_audio),
                        language=language,
                        filename=stream_filename,
                        content_type=stream_content_type,
                    )
                    full_text = (result.text or "").strip()

                    if full_text:
                        current_words = full_text.split()
                        if previous_full_text:
                            shared_prefix = _shared_prefix_length(previous_full_text, full_text)
                            stable_word_count = max(shared_prefix, committed_word_count)
                        else:
                            stable_word_count = 0
                        incremental_text = " ".join(
                            current_words[committed_word_count:stable_word_count]
                        ).strip()
                        await websocket.send_json({
                            "type": "transcript",
                            "text": incremental_text,
                            "full_text": full_text,
                            "chunk_index": chunk_index,
                            "is_final": False,
                            "confidence": result.average_confidence,
                        })
                        committed_word_count = stable_word_count
                        previous_full_text = full_text
                    chunk_index += 1
                except Exception as e:
                    logger.error(f"STT chunk error: {e}")
                    await websocket.send_json({
                        "type": "error",
                        "message": str(e),
                        "chunk_index": chunk_index,
                    })
                continue

            if "bytes" in data:
                audio_bytes = data["bytes"]
                try:
                    # Browser MediaRecorder chunks are often partial container fragments.
                    # Transcribing each fragment independently can fail. Keep a cumulative
                    # buffer and transcribe that instead, then emit only new text.
                    cumulative_audio.extend(audio_bytes)
                    result = await stt.transcribe(
                        bytes(cumulative_audio),
                        language=language,
                        filename=stream_filename,
                        content_type=stream_content_type,
                    )
                    full_text = (result.text or "").strip()

                    if full_text:
                        current_words = full_text.split()
                        if previous_full_text:
                            shared_prefix = _shared_prefix_length(previous_full_text, full_text)
                            stable_word_count = max(shared_prefix, committed_word_count)
                        else:
                            stable_word_count = 0
                        incremental_text = " ".join(
                            current_words[committed_word_count:stable_word_count]
                        ).strip()
                        logger.info(
                            "WS transcript payload chunk=%s incremental=%r full=%r",
                            chunk_index,
                            incremental_text,
                            full_text,
                        )
                        await websocket.send_json({
                            "type": "transcript",
                            "text": incremental_text,
                            "full_text": full_text,
                            "chunk_index": chunk_index,
                            "is_final": False,
                            "confidence": result.average_confidence,
                        })
                        committed_word_count = stable_word_count
                        previous_full_text = full_text
                    chunk_index += 1
                except Exception as e:
                    logger.error(f"STT chunk error: {e}")
                    await websocket.send_json({
                        "type": "error",
                        "message": str(e),
                        "chunk_index": chunk_index,
                    })

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.close()
        except Exception:
            pass
