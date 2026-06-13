"""
IntelliVoice — Memory Document Schemas

Pydantic models for session-scoped conversation memory.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ConversationTurn(BaseModel):
    """A single turn in a conversation."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: float = Field(default_factory=time.time)
    emotion: Optional[str] = None
    language: Optional[str] = None
    audio_duration_s: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class SessionMemory(BaseModel):
    """Short-term memory for the current conversation session."""
    session_id: str
    user_id: Optional[str] = None
    turns: List[ConversationTurn] = Field(default_factory=list)
    created_at: float = Field(default_factory=time.time)
    last_activity: float = Field(default_factory=time.time)
    detected_language: Optional[str] = None
    summary: Optional[str] = None

    def add_turn(self, role: str, content: str, **kwargs) -> None:
        """Add a conversation turn."""
        self.turns.append(ConversationTurn(role=role, content=content, **kwargs))
        self.last_activity = time.time()

    def get_recent_turns(self, n: int = 10) -> List[ConversationTurn]:
        """Get the most recent N turns."""
        return self.turns[-n:]

    def to_messages(self, n: int = 10) -> List[Dict[str, str]]:
        """Convert recent turns to LLM message format."""
        return [
            {"role": t.role, "content": t.content}
            for t in self.get_recent_turns(n)
        ]
