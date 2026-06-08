"""
IntelliVoice — Session Manager

Manages WebSocket sessions, per-session state, and audio buffers.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from fastapi import WebSocket

from config.logging_config import get_logger

logger = get_logger("session_manager")


@dataclass
class SessionState:
    """State for a single session (WebSocket or REST)."""
    session_id: str
    websocket: Optional[WebSocket] = None
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    is_active: bool = True

    # Audio buffer
    audio_buffer: bytearray = field(default_factory=bytearray)
    is_speaking: bool = False
    speech_start_time: Optional[float] = None

    # Conversation state
    conversation_history: list = field(default_factory=list)
    speaker_embedding: Optional[Any] = None
    user_id: Optional[str] = None

    # Processing flags
    is_processing: bool = False

    @property
    def duration_s(self) -> float:
        return time.time() - self.created_at

    @property
    def audio_duration_s(self) -> float:
        """Estimate audio duration from buffer size (16kHz, 16-bit mono)."""
        return len(self.audio_buffer) / (16000 * 2)

    def reset_audio_buffer(self) -> None:
        """Clear the audio buffer."""
        self.audio_buffer = bytearray()
        self.is_speaking = False
        self.speech_start_time = None

    def append_audio(self, chunk: bytes) -> None:
        """Append audio data to the buffer."""
        self.audio_buffer.extend(chunk)
        self.last_activity = time.time()

    def add_to_history(self, role: str, content: str, metadata: Optional[Dict] = None) -> None:
        """Add a message to conversation history."""
        entry = {
            "role": role,
            "content": content,
            "timestamp": time.time(),
        }
        if metadata:
            entry["metadata"] = metadata
        self.conversation_history.append(entry)

    async def send_json(self, message: dict) -> bool:
        """Send JSON message if WebSocket is available."""
        if self.websocket is not None:
            try:
                await self.websocket.send_json(message)
                return True
            except Exception:
                return False
        return False


class SessionManager:
    """Manages all active WebSocket sessions."""

    def __init__(self):
        self._sessions: Dict[str, SessionState] = {}
        self._lock = asyncio.Lock()

    async def create_session(self, websocket: WebSocket) -> SessionState:
        """Create and register a new session."""
        session_id = str(uuid.uuid4())[:8]
        session = SessionState(session_id=session_id, websocket=websocket)

        async with self._lock:
            self._sessions[session_id] = session

        logger.info("session_created", session_id=session_id)
        return session

    async def remove_session(self, session_id: str) -> None:
        """Remove a session."""
        async with self._lock:
            session = self._sessions.pop(session_id, None)
            if session:
                session.is_active = False
                logger.info(
                    "session_removed",
                    session_id=session_id,
                    duration_s=f"{session.duration_s:.1f}",
                    messages=len(session.conversation_history),
                )

    def get_session(self, session_id: str) -> Optional[SessionState]:
        """Get a session by ID."""
        return self._sessions.get(session_id)

    @property
    def active_count(self) -> int:
        return len(self._sessions)

    async def broadcast(self, message: dict) -> None:
        """Send a message to all active sessions."""
        disconnected = []
        for session_id, session in self._sessions.items():
            try:
                await session.websocket.send_json(message)
            except Exception:
                disconnected.append(session_id)

        for sid in disconnected:
            await self.remove_session(sid)

    async def cleanup_stale(self, max_idle_s: int = 300) -> None:
        """Remove sessions idle for too long."""
        now = time.time()
        stale = [
            sid for sid, s in self._sessions.items()
            if (now - s.last_activity) > max_idle_s
        ]
        for sid in stale:
            logger.warning("stale_session_cleanup", session_id=sid)
            await self.remove_session(sid)
