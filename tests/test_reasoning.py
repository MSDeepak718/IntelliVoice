"""
Tests for the Reasoning and Memory layers.
"""

import pytest
import time

from backend.layers.memory.schemas import SessionMemory, ConversationTurn, UserProfile
from backend.layers.memory.conversation_memory import ConversationMemory
from backend.layers.response_planning.planner import ResponsePlanner, ResponsePlan


class TestSessionMemory:
    """Tests for SessionMemory schema."""

    def test_create_session(self):
        session = SessionMemory(session_id="test-1")
        assert session.session_id == "test-1"
        assert len(session.turns) == 0

    def test_add_turn(self):
        session = SessionMemory(session_id="test-1")
        session.add_turn("user", "Hello")
        session.add_turn("assistant", "Hi there!")

        assert len(session.turns) == 2
        assert session.turns[0].role == "user"
        assert session.turns[1].content == "Hi there!"

    def test_emotion_tracking(self):
        session = SessionMemory(session_id="test-1")
        session.add_turn("user", "I'm upset", emotion="angry")
        session.add_turn("user", "Actually I'm fine", emotion="neutral")

        assert session.emotion_history == ["angry", "neutral"]

    def test_to_messages(self):
        session = SessionMemory(session_id="test-1")
        for i in range(20):
            session.add_turn("user", f"Message {i}")

        messages = session.to_messages(n=5)
        assert len(messages) == 5
        assert messages[0]["content"] == "Message 15"

    def test_recent_turns(self):
        session = SessionMemory(session_id="test-1")
        session.add_turn("user", "First")
        session.add_turn("assistant", "Second")
        session.add_turn("user", "Third")

        recent = session.get_recent_turns(2)
        assert len(recent) == 2
        assert recent[0].content == "Second"


class TestUserProfile:
    """Tests for UserProfile schema."""

    def test_create_profile(self):
        profile = UserProfile(user_id="user-1", name="Test User")
        assert profile.user_id == "user-1"
        assert profile.total_sessions == 0

    def test_update_last_seen(self):
        profile = UserProfile(user_id="user-1")
        old_seen = profile.last_seen
        time.sleep(0.01)
        profile.update_last_seen()
        assert profile.last_seen > old_seen
        assert profile.total_sessions == 1


class TestConversationMemory:
    """Tests for ConversationMemory."""

    @pytest.fixture
    def memory(self):
        return ConversationMemory()

    @pytest.mark.asyncio
    async def test_initialize(self, memory):
        await memory.initialize()

    def test_create_session(self, memory):
        session = memory.create_session("s1")
        assert session.session_id == "s1"
        assert memory.active_sessions == 1

    def test_add_and_get_turn(self, memory):
        memory.create_session("s1")
        memory.add_turn("s1", "user", "Hello", emotion="happy")

        context = memory.get_conversation_context("s1")
        assert len(context) == 1
        assert context[0]["content"] == "Hello"

    def test_emotion_trend(self, memory):
        memory.create_session("s1")
        memory.add_turn("s1", "user", "Hi", emotion="happy")
        memory.add_turn("s1", "user", "Oh no", emotion="sad")

        trend = memory.get_emotion_trend("s1")
        assert trend == ["happy", "sad"]

    def test_remove_session(self, memory):
        memory.create_session("s1")
        memory.remove_session("s1")
        assert memory.active_sessions == 0


class TestResponsePlanner:
    """Tests for ResponsePlanner."""

    @pytest.fixture
    def planner(self):
        return ResponsePlanner()

    def test_basic_plan(self, planner):
        plan = planner.plan(
            response_text="Hello! How can I help you?",
            user_emotion="neutral",
        )
        assert isinstance(plan, ResponsePlan)
        assert plan.response_text == "Hello! How can I help you?"
        assert plan.language == "english"

    def test_empathetic_response_for_sad_user(self, planner):
        plan = planner.plan(
            response_text="I understand how you feel.",
            user_emotion="sad",
        )
        assert plan.emotion == "warm"
        assert plan.tone == "gentle"
        assert plan.speaking_rate < 1.0  # Slower for sad users

    def test_energetic_for_happy_user(self, planner):
        plan = planner.plan(
            response_text="That's great news!",
            user_emotion="happy",
        )
        assert plan.speaking_rate >= 1.0

    def test_early_conversation_warmth(self, planner):
        plan = planner.plan(
            response_text="Welcome!",
            user_emotion="neutral",
            conversation_context=[{"role": "user", "content": "Hi"}],
        )
        assert plan.tone == "warm"

    def test_to_dict(self, planner):
        plan = planner.plan("Test response")
        d = plan.to_dict()
        assert "intent" in d
        assert "emotion" in d
        assert "response" in d
