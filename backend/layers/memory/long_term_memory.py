"""
IntelliVoice — Long-Term Memory (MongoDB)

Persists user profiles, conversation summaries, and preferences
to MongoDB for cross-session continuity.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from config import get_settings
from config.logging_config import get_logger
from backend.layers.memory.schemas import UserProfile

logger = get_logger("long_term_memory")


class LongTermMemory:
    """
    MongoDB-backed long-term memory store.

    Stores:
        - User profiles and preferences
        - Conversation summaries
        - Speaker embeddings for identification
        - Past interaction patterns
    """

    def __init__(self):
        self._client = None
        self._db = None
        self._is_connected = False

    async def connect(self) -> None:
        """Connect to MongoDB."""
        settings = get_settings()
        try:
            from motor.motor_asyncio import AsyncIOMotorClient

            self._client = AsyncIOMotorClient(settings.mongodb_uri)
            self._db = self._client[settings.mongodb_db_name]

            # Test connection
            await self._client.admin.command("ping")
            self._is_connected = True

            # Create indexes
            await self._ensure_indexes()

            logger.info("mongodb_connected", db=settings.mongodb_db_name)
        except Exception as e:
            logger.warning("mongodb_connection_failed", error=str(e), fallback="in_memory")
            self._is_connected = False

    async def _ensure_indexes(self) -> None:
        """Create necessary indexes."""
        if not self._is_connected:
            return

        # User profiles
        await self._db.user_profiles.create_index("user_id", unique=True)
        await self._db.user_profiles.create_index("last_seen")

        # Conversation logs
        await self._db.conversations.create_index("session_id", unique=True)
        await self._db.conversations.create_index("user_id")
        await self._db.conversations.create_index("created_at")

    async def save_user_profile(self, profile: UserProfile) -> None:
        """Save or update a user profile."""
        if not self._is_connected:
            return

        await self._db.user_profiles.update_one(
            {"user_id": profile.user_id},
            {"$set": profile.model_dump()},
            upsert=True,
        )
        logger.debug("user_profile_saved", user_id=profile.user_id)

    async def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Retrieve a user profile."""
        if not self._is_connected:
            return None

        doc = await self._db.user_profiles.find_one({"user_id": user_id})
        if doc:
            doc.pop("_id", None)
            return UserProfile(**doc)
        return None

    async def save_conversation(
        self,
        session_id: str,
        user_id: Optional[str],
        turns: List[Dict],
        summary: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> None:
        """Save a conversation to MongoDB."""
        if not self._is_connected:
            return

        # Serialize turns
        serialized_turns = []
        for t in turns:
            if hasattr(t, "model_dump"):
                serialized_turns.append(t.model_dump())
            elif hasattr(t, "dict"):
                serialized_turns.append(t.dict())
            elif hasattr(t, "__dict__"):
                serialized_turns.append(t.__dict__)
            else:
                serialized_turns.append(t)

        doc = {
            "session_id": session_id,
            "user_id": user_id,
            "turns": serialized_turns,
            "summary": summary,
            "metadata": metadata or {},
            "created_at": time.time(),
            "turn_count": len(turns),
        }

        await self._db.conversations.update_one(
            {"session_id": session_id},
            {"$set": doc},
            upsert=True,
        )
        logger.debug("conversation_saved", session_id=session_id, turns=len(turns))

    async def get_user_conversations(
        self,
        user_id: str,
        limit: int = 10,
    ) -> List[Dict]:
        """Get recent conversations for a user."""
        if not self._is_connected:
            return []

        cursor = self._db.conversations.find(
            {"user_id": user_id}
        ).sort("created_at", -1).limit(limit)

        conversations = []
        async for doc in cursor:
            doc.pop("_id", None)
            conversations.append(doc)

        return conversations

    async def search_conversations(
        self,
        user_id: str,
        query: str,
        limit: int = 5,
    ) -> List[Dict]:
        """Search past conversations by text content."""
        if not self._is_connected:
            return []

        # Simple text search (could use MongoDB Atlas Search for better results)
        cursor = self._db.conversations.find({
            "user_id": user_id,
            "$text": {"$search": query},
        }).limit(limit)

        results = []
        async for doc in cursor:
            doc.pop("_id", None)
            results.append(doc)

        return results

    async def disconnect(self) -> None:
        """Disconnect from MongoDB."""
        if self._client:
            self._client.close()
            self._is_connected = False
            logger.info("mongodb_disconnected")

    @property
    def is_connected(self) -> bool:
        return self._is_connected
