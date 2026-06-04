"""
IntelliVoice — Conversation Memory (LangGraph)

Layer 7: Manages conversation state using LangGraph for stateful
multi-turn dialogue management with tool orchestration.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, TypedDict, Annotated

from config.logging_config import get_logger
from backend.layers.memory.schemas import SessionMemory, ConversationTurn

logger = get_logger("conversation_memory")


# LangGraph state definition
class ConversationState(TypedDict):
    """State maintained across the conversation graph."""
    session_id: str
    messages: List[Dict[str, str]]
    current_emotion: str
    current_language: str
    user_intent: str
    speaker_embedding: Optional[List[float]]
    response_plan: Optional[Dict[str, str]]
    audio_response: Optional[bytes]
    turn_count: int
    is_complete: bool


class ConversationMemory:
    """
    Conversation memory manager using LangGraph.

    Manages:
        - Session memory (current conversation)
        - State transitions
        - Conversation graph execution
    """

    def __init__(self):
        self._sessions: Dict[str, SessionMemory] = {}
        self._graph = None
        self._is_initialized = False

    async def initialize(self) -> None:
        """Initialize the conversation graph."""
        try:
            self._build_graph()
            self._is_initialized = True
            logger.info("conversation_memory_initialized")
        except ImportError:
            logger.warning(
                "langgraph_not_available",
                fallback="simple_memory",
            )
            self._is_initialized = True

    def _build_graph(self) -> None:
        """Build the LangGraph conversation state machine."""
        try:
            from langgraph.graph import StateGraph, END

            graph = StateGraph(ConversationState)

            # Define nodes
            graph.add_node("process_input", self._process_input_node)
            graph.add_node("understand", self._understand_node)
            graph.add_node("plan_response", self._plan_response_node)
            graph.add_node("generate_response", self._generate_response_node)

            # Define edges
            graph.set_entry_point("process_input")
            graph.add_edge("process_input", "understand")
            graph.add_edge("understand", "plan_response")
            graph.add_edge("plan_response", "generate_response")
            graph.add_edge("generate_response", END)

            self._graph = graph.compile()
            logger.info("conversation_graph_built")
        except ImportError:
            logger.warning("langgraph_import_failed", using="simple_state_machine")
            self._graph = None

    async def _process_input_node(self, state: ConversationState) -> ConversationState:
        """Process and validate input."""
        state["turn_count"] = state.get("turn_count", 0) + 1
        return state

    async def _understand_node(self, state: ConversationState) -> ConversationState:
        """Understand the user's message (filled by pipeline)."""
        return state

    async def _plan_response_node(self, state: ConversationState) -> ConversationState:
        """Plan the response (filled by pipeline)."""
        return state

    async def _generate_response_node(self, state: ConversationState) -> ConversationState:
        """Generate the response (filled by pipeline)."""
        return state

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

    async def get_state(self, session_id: str) -> ConversationState:
        """Get the current conversation state."""
        session = self._sessions.get(session_id)
        messages = session.to_messages() if session else []

        return ConversationState(
            session_id=session_id,
            messages=messages,
            current_emotion=session.emotion_history[-1] if session and session.emotion_history else "neutral",
            current_language=session.detected_language or "english",
            user_intent="unknown",
            speaker_embedding=session.speaker_embedding if session else None,
            response_plan=None,
            audio_response=None,
            turn_count=len(session.turns) if session else 0,
            is_complete=False,
        )

    def remove_session(self, session_id: str) -> None:
        """Remove a session."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info("session_memory_removed", session_id=session_id)

    @property
    def active_sessions(self) -> int:
        return len(self._sessions)
