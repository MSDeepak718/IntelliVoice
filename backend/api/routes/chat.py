"""
IntelliVoice — Chat REST API Route

REST endpoint for text-based chat (non-streaming).
"""

from __future__ import annotations

import time
from typing import Optional

from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel, Field

from config.logging_config import get_logger
from backend.core.session_manager import SessionManager, SessionState

logger = get_logger("chat_api")
router = APIRouter()

# Reuse session manager for REST sessions
_rest_sessions: dict = {}


class ChatRequest(BaseModel):
    """Chat request body."""
    message: str = Field(..., min_length=1, max_length=2000)
    session_id: Optional[str] = None
    language: Optional[str] = None


class ChatResponse(BaseModel):
    """Chat response body."""
    response: str
    session_id: str
    metadata: dict = {}


@router.post("/chat", response_model=ChatResponse)
async def chat(request: Request, body: ChatRequest):
    """
    Text-based chat endpoint.

    Processes text input through the reasoning pipeline
    without audio processing layers.
    """
    pipeline = request.app.state.pipeline
    start_time = time.time()

    # Get or create session
    session_id = body.session_id or f"rest_{int(time.time())}"

    if session_id not in _rest_sessions:
        # Create a mock session state for REST usage
        _rest_sessions[session_id] = SessionState(
            session_id=session_id,
            websocket=None,  # No WebSocket for REST
        )
        pipeline.conversation_memory.create_session(session_id)

    session = _rest_sessions[session_id]

    try:
        result = await pipeline.process_text(
            text=body.message,
            session=session,
        )

        return ChatResponse(
            response=result["response_text"],
            session_id=session_id,
            metadata={
                **result.get("metadata", {}),
                "total_ms": round((time.time() - start_time) * 1000),
            },
        )

    except Exception as e:
        logger.error("chat_error", session=session_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/chat/{session_id}")
async def delete_session(request: Request, session_id: str):
    """Delete a chat session."""
    pipeline = request.app.state.pipeline

    if session_id in _rest_sessions:
        del _rest_sessions[session_id]
    pipeline.conversation_memory.remove_session(session_id)

    return {"status": "deleted", "session_id": session_id}
