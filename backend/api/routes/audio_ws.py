"""
IntelliVoice — WebSocket Audio Streaming Route

Main real-time audio endpoint. Handles bidirectional audio streaming
between the browser and the processing pipeline.

Protocol:
    Client → Server: Binary audio chunks (PCM int16, 16kHz, mono)
    Server → Client: JSON messages with transcription, response audio, and metadata
"""

from __future__ import annotations

import asyncio
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

                # Check VAD on the last 0.5 seconds of audio
                chunk_len = int(0.5 * sr)
                if waveform.shape[-1] >= chunk_len:
                    recent_waveform = waveform[..., -chunk_len:]
                else:
                    recent_waveform = waveform

                is_speech = pipeline.vad.is_speech(recent_waveform)

                if is_speech:
                    if not session.is_speaking:
                        session.is_speaking = True
                        session.speech_start_time = time.time()
                        await websocket.send_json({
                            "type": "vad",
                            "status": "speech_start",
                        })
                        
                        # Tell frontend to stop playing audio immediately
                        await websocket.send_json({
                            "type": "interrupt",
                            "status": "user_spoke",
                        })
                    
                    # Interruption logic: if assistant is processing in backend, cancel it
                    if session.is_processing and session.processing_task and not session.processing_task.done():
                        session.processing_task.cancel()
                        pipeline.cancel_processing()
                        session.is_processing = False
                        
                    session.silence_start_time = None
                    continue

                # Speech ended — check silence duration
                if session.is_speaking:
                    if session.silence_start_time is None:
                        session.silence_start_time = time.time()
                        
                    silence_duration = time.time() - session.silence_start_time
                    if silence_duration >= 1.2:  # Wait 1.2s to finish completely
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

                        # Create background processing task for streaming
                        buffer_copy = bytes(session.audio_buffer)
                        session.reset_audio_buffer()
                        
                        async def process_task(audio_bytes, session_ref):
                            try:
                                async for chunk in pipeline.stream_process_audio(
                                    audio_bytes=audio_bytes,
                                    session=session_ref,
                                    source_sample_rate=settings.sample_rate,
                                ):
                                    if chunk["type"] == "transcription":
                                        await websocket.send_json({
                                            "type": "transcription",
                                            "text": chunk["text"],
                                            "emotion": chunk.get("emotion", "neutral"),
                                            "language": chunk.get("language", "unknown"),
                                        })
                                    elif chunk["type"] == "response_start":
                                        await websocket.send_json({
                                            "type": "response_start"
                                        })
                                    elif chunk["type"] == "response_chunk":
                                        await websocket.send_json({
                                            "type": "response_chunk",
                                            "text": chunk["text"]
                                        })
                                    elif chunk["type"] == "response_audio":
                                        audio_b64 = base64.b64encode(chunk["audio_bytes"]).decode("utf-8")
                                        await websocket.send_json({
                                            "type": "audio_response",
                                            "audio": audio_b64,
                                            "sample_rate": chunk.get("sample_rate", 22050),
                                            "format": "pcm_int16_mono",
                                        })
                                    elif chunk["type"] == "error":
                                        await websocket.send_json({"type": "error", "message": chunk["message"]})
                            except asyncio.CancelledError:
                                logger.info("processing_task_cancelled", session=session_ref.session_id)
                            except Exception as e:
                                logger.error("pipeline_processing_error", error=str(e))
                                await websocket.send_json({"type": "error", "message": "Processing failed."})
                            finally:
                                session_ref.is_processing = False
                                await websocket.send_json({"type": "processing", "status": "complete"})

                        session.processing_task = asyncio.create_task(process_task(buffer_copy, session))

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
                        pipeline.memory.remove_session(session.session_id)
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
