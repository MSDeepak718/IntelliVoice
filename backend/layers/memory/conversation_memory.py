"""
IntelliVoice — Conversation Memory

Manages session memory for multi-turn dialogue.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from config.logging_config import get_logger
from backend.layers.memory.schemas import SessionMemory, ConversationTurn

logger = get_logger("conversation_memory")


class ConversationMemory:
    """
    Conversation memory manager.

    Manages:
        - Session memory (current conversation)
        - Turn history
    """

    def __init__(self):
        self._sessions: Dict[str, SessionMemory] = {}
        self._is_initialized = False

    async def initialize(self) -> None:
        """Initialize the conversation memory."""
        self._is_initialized = True
        logger.info("conversation_memory_initialized")

    def create_session(self, session_id: str, user_id: Optional[str] = None) -> SessionMemory:
        """Create a new session memory."""
        session = SessionMemory(session_id=session_id, user_id=user_id)
        self._sessions[session_id] = session
        logger.info("session_memory_created", session_id=session_id)
        return session

    def get_session(self, session_id: str) -> Optional[SessionMemory]:
        """Get session memory by ID."""
        return self._sessions.get(session_id)

    def add_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        emotion: Optional[str] = None,
        language: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Add a turn to session memory."""
        session = self._sessions.get(session_id)
        if not session:
            session = self.create_session(session_id)

        session.add_turn(
            role=role,
            content=content,
            emotion=emotion,
            language=language,
            **kwargs,
        )

        logger.debug(
            "turn_added",
            session_id=session_id,
            role=role,
            turns=len(session.turns),
        )

    def get_conversation_context(
        self,
        session_id: str,
        max_turns: int = 10,
    ) -> List[Dict[str, str]]:
        """Get conversation context for LLM input."""
        session = self._sessions.get(session_id)
        if not session:
            return []
        return session.to_messages(max_turns)

    def get_emotion_trend(self, session_id: str) -> List[str]:
        """Get the emotion trend for a session."""
        session = self._sessions.get(session_id)
        if not session:
            return []
        return session.emotion_history

    def remove_session(self, session_id: str) -> None:
        """Remove a session."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info("session_memory_removed", session_id=session_id)

    @property
    def active_sessions(self) -> int:
        return len(self._sessions)
