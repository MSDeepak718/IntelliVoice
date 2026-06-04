"""
IntelliVoice — WebSocket Audio Streaming Route

Main real-time audio endpoint. Handles bidirectional audio streaming
between the browser and the processing pipeline.

Protocol:
    Client → Server: Binary audio chunks (PCM int16, 16kHz, mono)
    Server → Client: JSON messages with transcription, response audio, and metadata
"""

from __future__ import annotations

import base64
import json
import time

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Request

from config import get_settings
from config.logging_config import get_logger
from backend.core.session_manager import SessionManager, SessionState
from backend.layers.preprocessing.audio_utils import get_audio_duration

logger = get_logger("ws_audio")
router = APIRouter()

# Global session manager
session_manager = SessionManager()


@router.websocket("/ws/audio")
async def audio_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for real-time audio streaming.

    Protocol:
        1. Client connects and sends config JSON
        2. Client streams binary audio chunks
        3. Server processes and responds with JSON:
            - type: "vad" — voice activity status
            - type: "transcription" — speech transcription
            - type: "response" — text response + audio
            - type: "error" — error message
    """
    await websocket.accept()
    settings = get_settings()

    # Create session
    session = await session_manager.create_session(websocket)
    pipeline = websocket.app.state.pipeline

    logger.info("ws_connected", session_id=session.session_id)

    # Send welcome message
    await websocket.send_json({
        "type": "connected",
        "session_id": session.session_id,
        "config": {
            "sample_rate": settings.sample_rate,
            "chunk_size_ms": settings.chunk_size_ms,
            "format": "pcm_int16_mono",
        },
    })

    try:
        while True:
            # Receive message (binary audio or JSON control)
            message = await websocket.receive()

            if "bytes" in message:
                # Binary audio data
                audio_chunk = message["bytes"]
                session.append_audio(audio_chunk)

                # Check if we have enough audio to process
                audio_duration = session.audio_duration_s
                if audio_duration < 0.5:
                    # Not enough audio yet, keep buffering
                    continue

                # Check VAD
                from backend.layers.preprocessing.audio_utils import bytes_to_waveform
                waveform, sr = bytes_to_waveform(bytes(session.audio_buffer))

                is_speech = pipeline.vad.is_speech(waveform)

                if is_speech:
                    if not session.is_speaking:
                        session.is_speaking = True
                        session.speech_start_time = time.time()
                        await websocket.send_json({
                            "type": "vad",
                            "status": "speech_start",
                        })
                    continue

                # Speech ended — process the buffered audio
                if session.is_speaking and audio_duration >= 0.5:
                    session.is_speaking = False
                    session.is_processing = True

                    await websocket.send_json({
                        "type": "vad",
                        "status": "speech_end",
                        "duration_s": round(audio_duration, 2),
                    })

                    await websocket.send_json({
                        "type": "processing",
                        "status": "started",
                    })

                    # Process through the full pipeline
                    try:
                        result = await pipeline.process_audio(
                            audio_bytes=bytes(session.audio_buffer),
                            session=session,
                            source_sample_rate=settings.sample_rate,
                        )

                        # Send transcription
                        if result.get("transcription"):
                            await websocket.send_json({
                                "type": "transcription",
                                "text": result["transcription"],
                                "emotion": result.get("emotion", "neutral"),
                                "language": result.get("metadata", {}).get("language", "unknown"),
                            })

                        # Send response text
                        if result.get("response_text"):
                            await websocket.send_json({
                                "type": "response",
                                "text": result["response_text"],
                                "metadata": result.get("metadata", {}),
                            })

                        # Send response audio (base64 encoded for WebSocket)
                        if result.get("response_audio"):
                            audio_b64 = base64.b64encode(
                                result["response_audio"]
                            ).decode("utf-8")

                            await websocket.send_json({
                                "type": "audio_response",
                                "audio": audio_b64,
                                "sample_rate": result.get("response_sample_rate", 22050),
                                "format": "pcm_int16_mono",
                            })

                    except Exception as e:
                        logger.error(
                            "pipeline_processing_error",
                            session=session.session_id,
                            error=str(e),
                        )
                        await websocket.send_json({
                            "type": "error",
                            "message": "Processing failed. Please try again.",
                        })

                    finally:
                        session.reset_audio_buffer()
                        session.is_processing = False

                        await websocket.send_json({
                            "type": "processing",
                            "status": "complete",
                        })

            elif "text" in message:
                # JSON control message
                try:
                    data = json.loads(message["text"])
                    msg_type = data.get("type", "")

                    if msg_type == "config":
                        # Client sending configuration
                        logger.info(
                            "client_config",
                            session=session.session_id,
                            config=data,
                        )

                    elif msg_type == "text_message":
                        # Text chat mode
                        text = data.get("text", "")
                        if text:
                            result = await pipeline.process_text(
                                text=text,
                                session=session,
                            )
                            await websocket.send_json({
                                "type": "response",
                                "text": result["response_text"],
                                "metadata": result.get("metadata", {}),
                            })

                    elif msg_type == "reset":
                        # Reset session
                        session.reset_audio_buffer()
                        session.conversation_history.clear()
                        pipeline.conversation_memory.remove_session(session.session_id)
                        await websocket.send_json({
                            "type": "reset",
                            "status": "ok",
                        })

                    elif msg_type == "ping":
                        await websocket.send_json({"type": "pong"})

                except json.JSONDecodeError:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Invalid JSON message",
                    })

    except WebSocketDisconnect:
        logger.info(
            "ws_disconnected",
            session_id=session.session_id,
            duration_s=round(session.duration_s, 1),
        )
    except Exception as e:
        logger.error(
            "ws_error",
            session_id=session.session_id,
            error=str(e),
        )
    finally:
        await session_manager.remove_session(session.session_id)
