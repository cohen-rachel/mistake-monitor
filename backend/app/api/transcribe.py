"""Transcription endpoints: REST and WebSocket streaming."""

import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, UploadFile, File, Form

from app.services.stt.factory import get_stt_provider

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/transcribe", tags=["transcribe"])


@router.post("")
async def transcribe_audio(
    audio_file: UploadFile = File(...),
    language: str = Form("en"),
):
    """Transcribe an uploaded audio file and return the transcript."""
    audio_bytes = await audio_file.read()
    stt = get_stt_provider()
    result = await stt.transcribe(audio_bytes, language)

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


@router.websocket("/stream")
async def transcribe_stream(websocket: WebSocket):
    """WebSocket endpoint for real-time audio transcription.

    Client sends binary audio chunks. Server responds with JSON transcription results.
    Send a text message 'stop' to end the session.
    """
    await websocket.accept()
    stt = get_stt_provider()
    language = websocket.query_params.get("language", "en")
    chunk_index = 0
    cumulative_audio = bytearray()

    try:
        while True:
            # Receive data — could be binary (audio) or text (control)
            data = await websocket.receive()

            if data.get("type") == "websocket.disconnect":
                break

            if "text" in data:
                text_msg = data["text"]
                if text_msg.strip().lower() == "stop":
                    await websocket.send_json({
                        "type": "stopped",
                        "chunk_index": chunk_index,
                    })
                    break
                continue

            if "bytes" in data:
                audio_bytes = data["bytes"]
                try:
                    # Browser MediaRecorder chunks are often partial container fragments.
                    # Transcribing each fragment independently can fail. Keep a cumulative
                    # buffer and transcribe that instead, then emit only new text.
                    cumulative_audio.extend(audio_bytes)
                    result = await stt.transcribe(bytes(cumulative_audio), language=language)
                    full_text = (result.text or "").strip()

                    if full_text:
                        await websocket.send_json({
                            "type": "transcript",
                            "text": full_text,
                            "full_text": full_text,
                            "chunk_index": chunk_index,
                            "is_final": False,
                            "confidence": result.average_confidence,
                        })
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
