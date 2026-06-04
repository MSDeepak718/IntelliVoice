"""
IntelliVoice — Memory Document Schemas

Pydantic models for conversation memory documents stored in MongoDB.
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
    speaker_embedding: Optional[List[float]] = None
    detected_language: Optional[str] = None
    emotion_history: List[str] = Field(default_factory=list)
    summary: Optional[str] = None

    def add_turn(self, role: str, content: str, **kwargs) -> None:
        """Add a conversation turn."""
        self.turns.append(ConversationTurn(role=role, content=content, **kwargs))
        self.last_activity = time.time()
        if "emotion" in kwargs and kwargs["emotion"]:
            self.emotion_history.append(kwargs["emotion"])

    def get_recent_turns(self, n: int = 10) -> List[ConversationTurn]:
        """Get the most recent N turns."""
        return self.turns[-n:]

    def to_messages(self, n: int = 10) -> List[Dict[str, str]]:
        """Convert recent turns to LLM message format."""
        return [
            {"role": t.role, "content": t.content}
            for t in self.get_recent_turns(n)
        ]


class UserProfile(BaseModel):
    """Long-term user profile stored in MongoDB."""
    user_id: str
    name: Optional[str] = None
    preferred_language: str = "english"
    preferences: Dict[str, Any] = Field(default_factory=dict)
    speaker_embeddings: List[List[float]] = Field(default_factory=list)
    total_sessions: int = 0
    total_turns: int = 0
    created_at: float = Field(default_factory=time.time)
    last_seen: float = Field(default_factory=time.time)
    conversation_summaries: List[str] = Field(default_factory=list)

    def update_last_seen(self) -> None:
        self.last_seen = time.time()
        self.total_sessions += 1
