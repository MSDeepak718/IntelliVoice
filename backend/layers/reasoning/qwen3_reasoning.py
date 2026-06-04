"""
IntelliVoice — Qwen3 30B-A3B MoE Reasoning Layer

Layer 6: Core reasoning engine using Qwen3 MoE (30B total, 3B active).
Handles intent understanding, planning, question answering, and dialogue management.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from config import get_settings
from config.logging_config import get_logger
from config.model_registry import ModelRegistry

logger = get_logger("qwen3_reasoning")

SYSTEM_PROMPT = """You are IntelliVoice, a multilingual AI voice assistant. You communicate naturally and empathetically.

Key behaviors:
- Respond in the same language the user speaks (English, Hindi, Tamil, Telugu, or code-mixed)
- Match the user's emotional tone appropriately
- Keep responses concise and conversational (suitable for speech output)
- Be helpful, warm, and professional
- If the user sounds distressed, be empathetic first before solving their problem
- Avoid long lists or complex formatting — your response will be spoken aloud

You receive context about the user's:
- Speech transcription
- Detected emotion and energy level
- Speaking style
- Conversation history

Use this context to generate appropriate, emotionally-aware responses."""


class Qwen3Reasoner:
    """
    Qwen3 30B-A3B MoE reasoning engine.

    Architecture: 30B parameters, 3B active per token (MoE).
    Loaded with INT4 quantization to fit within RTX 4080 VRAM.

    Responsibilities:
        - Intent understanding
        - Planning and problem solving
        - Question answering
        - Dialogue management
        - Emotion-aware response generation
    """

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self._device: torch.device = torch.device("cpu")
        self._is_loaded = False
        self._config = ModelRegistry.QWEN3_MOE
        self._settings = get_settings()

    async def load(self, device: torch.device = torch.device("cuda")) -> None:
        """Load Qwen3 MoE with INT4 quantization."""
        if self._is_loaded:
            return

        logger.info("loading_qwen3_moe", model=self._config.hf_model_id)

        try:
            quantization_config = None
            if device.type == "cuda":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )

            self.tokenizer = AutoTokenizer.from_pretrained(
                self._config.hf_model_id,
                trust_remote_code=True,
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                self._config.hf_model_id,
                device_map="auto" if device.type == "cuda" else None,
                quantization_config=quantization_config,
                trust_remote_code=True,
                torch_dtype=torch.float16,
            )
            self.model.eval()
            self._device = device
            self._is_loaded = True

            num_params = sum(p.numel() for p in self.model.parameters()) / 1e9
            logger.info("qwen3_moe_loaded", params_b=f"{num_params:.1f}", device=str(device))
        except Exception as e:
            logger.error("qwen3_moe_load_failed", error=str(e))
            raise

    @torch.inference_mode()
    def generate_response(
        self,
        user_message: str,
        conversation_history: List[Dict[str, str]] = None,
        emotion_context: Optional[Dict] = None,
        speaker_context: Optional[Dict] = None,
        system_prompt: str = SYSTEM_PROMPT,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Generate a reasoned response.

        Args:
            user_message: Transcribed user speech.
            conversation_history: Previous conversation turns.
            emotion_context: Detected emotion info from Emotion2Vec.
            speaker_context: Speaker characteristics from WavLM.
            system_prompt: System instruction.
            max_new_tokens: Max tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.

        Returns:
            Dict with 'response', 'tokens_used', and metadata.
        """
        if not self._is_loaded:
            raise RuntimeError("Qwen3 MoE not loaded.")

        max_new_tokens = max_new_tokens or self._settings.max_new_tokens
        temperature = temperature or self._settings.temperature
        top_p = top_p or self._settings.top_p

        # Build context-enriched system prompt
        enriched_system = system_prompt
        if emotion_context:
            enriched_system += f"\n\n[User Emotion Context]\n"
            enriched_system += f"- Emotion: {emotion_context.get('emotion', 'unknown')}\n"
            enriched_system += f"- Energy: {emotion_context.get('energy_level', 'unknown')}\n"
            enriched_system += f"- Speaking rate: {emotion_context.get('speaking_rate', 'unknown')}\n"
            enriched_system += f"- Confidence: {emotion_context.get('confidence', 0)}\n"

        # Build messages list
        messages = [{"role": "system", "content": enriched_system}]

        if conversation_history:
            for entry in conversation_history[-10:]:  # Last 10 turns
                messages.append({
                    "role": entry.get("role", "user"),
                    "content": entry.get("content", ""),
                })

        messages.append({"role": "user", "content": user_message})

        # Tokenize
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self._device)

        # Generate
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        # Decode only new tokens
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        result = {
            "response": response.strip(),
            "tokens_generated": len(new_tokens),
            "input_tokens": inputs["input_ids"].shape[1],
        }

        logger.debug(
            "response_generated",
            response_length=len(response),
            tokens=len(new_tokens),
        )

        return result

    @torch.inference_mode()
    def plan_response(
        self,
        transcription: str,
        emotion: str = "neutral",
        intent: str = "unknown",
        language: str = "english",
    ) -> Dict[str, str]:
        """
        Plan a response — determine intent, emotion, and tone before generating.

        Returns:
            Dict with 'intent', 'response_emotion', 'response_tone', 'planned_response'.
        """
        planning_prompt = (
            f"The user said: \"{transcription}\"\n"
            f"Their detected emotion: {emotion}\n"
            f"Their detected intent: {intent}\n"
            f"Language: {language}\n\n"
            "Plan your response. Output JSON with:\n"
            '- "response_intent": what you plan to do (answer/clarify/empathize/redirect)\n'
            '- "response_emotion": emotion to convey (warm/empathetic/professional/excited)\n'
            '- "response_tone": tone to use (casual/formal/gentle/energetic)\n'
            '- "response": your actual response text\n'
            "Respond ONLY with the JSON."
        )

        result = self.generate_response(
            user_message=planning_prompt,
            max_new_tokens=256,
            temperature=0.3,
        )

        try:
            parsed = json.loads(result["response"])
            return parsed
        except json.JSONDecodeError:
            return {
                "response_intent": "answer",
                "response_emotion": "neutral",
                "response_tone": "professional",
                "response": result["response"],
            }

    def offload_to_cpu(self) -> None:
        """Offload to CPU."""
        if self.model is not None:
            self.model = self.model.cpu()
            torch.cuda.empty_cache()
            self._device = torch.device("cpu")
            logger.info("qwen3_offloaded_to_cpu")

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded
