"""
IntelliVoice — Response Planning Layer

Layer 9: Plans the response before speech synthesis.
Determines intent, emotion, tone, and text to synthesize.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from config.logging_config import get_logger

logger = get_logger("response_planner")


@dataclass
class ResponsePlan:
    """Structured response plan for speech synthesis."""
    intent: str = "answer"          # answer, clarify, empathize, redirect, greet
    emotion: str = "neutral"        # warm, empathetic, professional, excited, calm
    tone: str = "conversational"    # casual, formal, gentle, energetic
    language: str = "english"       # Target output language
    response_text: str = ""         # Text to synthesize
    ssml_hints: Optional[str] = None  # Optional SSML for prosody control
    speaking_rate: float = 1.0      # 0.5 = slow, 1.0 = normal, 1.5 = fast
    pitch_shift: float = 0.0       # Semitones to shift pitch

    def to_dict(self) -> Dict:
        return {
            "intent": self.intent,
            "emotion": self.emotion,
            "tone": self.tone,
            "language": self.language,
            "response": self.response_text,
            "speaking_rate": self.speaking_rate,
            "pitch_shift": self.pitch_shift,
        }


# Emotion-to-tone mapping for consistent responses
EMOTION_TONE_MAP = {
    "angry": {"tone": "gentle", "emotion": "empathetic", "rate": 0.9},
    "sad": {"tone": "gentle", "emotion": "warm", "rate": 0.85},
    "happy": {"tone": "energetic", "emotion": "excited", "rate": 1.1},
    "frustrated": {"tone": "calm", "emotion": "empathetic", "rate": 0.9},
    "neutral": {"tone": "conversational", "emotion": "professional", "rate": 1.0},
    "fearful": {"tone": "gentle", "emotion": "reassuring", "rate": 0.85},
    "surprised": {"tone": "casual", "emotion": "engaged", "rate": 1.05},
}


class ResponsePlanner:
    """
    Plans structured responses before speech synthesis.

    Takes the LLM output and enriches it with appropriate
    emotion, tone, and delivery parameters based on the
    user's emotional state and conversation context.
    """

    def plan(
        self,
        response_text: str,
        user_emotion: str = "neutral",
        user_intent: str = "unknown",
        detected_language: str = "english",
        conversation_context: Optional[List[Dict]] = None,
    ) -> ResponsePlan:
        """
        Create a response plan.

        Args:
            response_text: Generated response text from LLM.
            user_emotion: Detected user emotion.
            user_intent: Detected user intent.
            detected_language: Detected/preferred language.
            conversation_context: Recent conversation turns.

        Returns:
            ResponsePlan with delivery parameters.
        """
        # Get emotion-appropriate delivery settings
        tone_settings = EMOTION_TONE_MAP.get(
            user_emotion,
            EMOTION_TONE_MAP["neutral"],
        )

        # Determine response intent
        response_intent = self._classify_response_intent(
            response_text, user_intent
        )

        plan = ResponsePlan(
            intent=response_intent,
            emotion=tone_settings["emotion"],
            tone=tone_settings["tone"],
            language=detected_language,
            response_text=response_text,
            speaking_rate=tone_settings["rate"],
        )

        # Adjust for context
        if conversation_context and len(conversation_context) <= 2:
            # Early in conversation — be more welcoming
            plan.tone = "warm"
            plan.emotion = "friendly"

        logger.debug(
            "response_planned",
            intent=plan.intent,
            emotion=plan.emotion,
            tone=plan.tone,
            language=plan.language,
            text_length=len(response_text),
        )

        return plan

    def _classify_response_intent(
        self,
        response_text: str,
        user_intent: str,
    ) -> str:
        """Classify the response intent."""
        text_lower = response_text.lower()

        if "?" in response_text:
            return "clarify"
        elif any(w in text_lower for w in ["sorry", "understand", "i see"]):
            return "empathize"
        elif any(w in text_lower for w in ["hello", "hi", "welcome", "hey"]):
            return "greet"
        elif user_intent == "question":
            return "answer"
        else:
            return "answer"

    def adjust_for_speaker(
        self,
        plan: ResponsePlan,
        speaker_characteristics: Optional[Dict] = None,
    ) -> ResponsePlan:
        """Adjust response plan based on speaker characteristics."""
        if not speaker_characteristics:
            return plan

        # If speaker speaks slowly, respond slightly slower
        if speaker_characteristics.get("speaking_rate") == "slow":
            plan.speaking_rate = min(plan.speaking_rate, 0.9)
        elif speaker_characteristics.get("speaking_rate") == "fast":
            plan.speaking_rate = max(plan.speaking_rate, 1.1)

        return plan
