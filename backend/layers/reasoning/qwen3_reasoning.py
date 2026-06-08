"""
IntelliVoice — Qwen3-14B Reasoning Layer

Layer 6: Core reasoning engine.

Model: Qwen/Qwen3-14B
    - 14B parameters, ~28GB FP16 → ~8.5GB NF4 (with double quant)
    - Strong multilingual + code reasoning
    - Long context window (32k tokens)
    - Fits comfortably in 16GB VRAM with 4-bit quantization

Responsibilities:
    - Intent understanding
    - Planning and problem solving
    - Question answering (with RAG context)
    - Dialogue management
    - Emotion-aware response generation
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from config import get_settings
from config.logging_config import get_logger
from config.model_registry import ModelRegistry

logger = get_logger("qwen3_reasoning")


SYSTEM_PROMPT = """You are IntelliVoice, a multilingual AI voice assistant. \
You communicate naturally and empathetically.

Key behaviors:
- Respond in the same language the user speaks (English, Hindi, Tamil, Telugu, or code-mixed)
- Match the user's emotional tone appropriately
- Keep responses concise and conversational (suitable for speech output, ideally 1-3 sentences)
- Be helpful, warm, and professional
- If the user sounds distressed, be empathetic first before solving their problem
- Avoid long lists, markdown, emojis, or code blocks — your response will be spoken aloud
- If the user's question is ambiguous, ask one short clarifying question

You receive context about the user's:
- Speech transcription
- Detected emotion and energy level
- Speaking style
- Recent conversation turns
- Optional retrieved knowledge from the knowledge base

Use this context to generate appropriate, emotionally-aware spoken responses."""


class Qwen3Reasoner:
    """
    Qwen3-14B reasoning engine (4-bit NF4 with double quant).
    """

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self._device: torch.device = torch.device("cpu")
        self._is_loaded = False
        self._config = ModelRegistry.QWEN3_14B
        self._settings = get_settings()

    async def load(self, device: torch.device = torch.device("cuda")) -> None:
        """Load Qwen3-14B with NF4 double-quant."""
        if self._is_loaded:
            return
        logger.info("loading_qwen3_14b", model=self._config.hf_model_id)

        try:
            quantization_config = None
            if device.type == "cuda" and self._config.quantization_config:
                quantization_config = BitsAndBytesConfig(**{
                    "load_in_4bit": self._config.quantization_config.get("load_in_4bit", True),
                    "bnb_4bit_compute_dtype": torch.float16,
                    "bnb_4bit_quant_type": self._config.quantization_config.get("bnb_4bit_quant_type", "nf4"),
                    "bnb_4bit_use_double_quant": self._config.quantization_config.get("bnb_4bit_use_double_quant", True),
                })

            self.tokenizer = AutoTokenizer.from_pretrained(
                self._config.hf_model_id,
                trust_remote_code=True,
            )
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            self.model = AutoModelForCausalLM.from_pretrained(
                self._config.hf_model_id,
                device_map="auto" if device.type == "cuda" else None,
                quantization_config=quantization_config,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                attn_implementation="sdpa",  # scaled-dot-product attention (no extra deps)
            )
            self.model.eval()
            self._device = device
            self._is_loaded = True

            num_params = sum(p.numel() for p in self.model.parameters()) / 1e9
            logger.info(
                "qwen3_14b_loaded",
                params_b=f"{num_params:.1f}",
                device=str(device),
                quant="nf4_double_quant",
            )
        except Exception as e:
            logger.error("qwen3_14b_load_failed", error=str(e))
            raise

    def _build_messages(
        self,
        user_message: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        emotion_context: Optional[Dict] = None,
        rag_context: str = "",
        system_prompt: str = SYSTEM_PROMPT,
    ) -> List[Dict[str, str]]:
        """Compose the chat-template messages list."""
        enriched_system = system_prompt

        if emotion_context:
            enriched_system += "\n\n[User emotion context]\n"
            enriched_system += f"- Emotion: {emotion_context.get('emotion', 'unknown')}\n"
            enriched_system += f"- Energy: {emotion_context.get('energy_level', 'unknown')}\n"
            enriched_system += f"- Speaking rate: {emotion_context.get('speaking_rate', 'unknown')}\n"
            confidence = emotion_context.get("confidence", 0)
            if isinstance(confidence, (int, float)) and confidence > 0:
                enriched_system += f"- Confidence: {confidence:.2f}\n"

        if rag_context:
            enriched_system += f"\n\n[Retrieved knowledge]\n{rag_context}"

        messages: List[Dict[str, str]] = [{"role": "system", "content": enriched_system}]

        if conversation_history:
            # Keep the most recent turns, drop empties
            trimmed = [m for m in conversation_history[-10:] if m.get("content")]
            messages.extend({"role": m["role"], "content": m["content"]} for m in trimmed)

        messages.append({"role": "user", "content": user_message})
        return messages

    @torch.inference_mode()
    def generate_response(
        self,
        user_message: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        emotion_context: Optional[Dict] = None,
        rag_context: str = "",
        system_prompt: str = SYSTEM_PROMPT,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Generate a reasoned, emotion-aware response.

        Args:
            user_message: Transcribed user speech.
            conversation_history: Previous turns (most recent last).
            emotion_context: Dict from Emotion2Vec.
            rag_context: Formatted RAG context (from `RAGRetriever.format_context`).
            system_prompt: Override the default system prompt.
            max_new_tokens: Cap on new tokens.
            temperature: Sampling temperature.
            top_p: Nucleus sampling.

        Returns:
            Dict with 'response', 'tokens_generated', 'input_tokens', 'finish_reason'.
        """
        if not self._is_loaded:
            raise RuntimeError("Qwen3-14B not loaded.")

        max_new_tokens = max_new_tokens or self._settings.max_new_tokens
        temperature = temperature if temperature is not None else self._settings.temperature
        top_p = top_p if top_p is not None else self._settings.top_p

        messages = self._build_messages(
            user_message=user_message,
            conversation_history=conversation_history,
            emotion_context=emotion_context,
            rag_context=rag_context,
            system_prompt=system_prompt,
        )

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self._device)

        gen_kwargs: Dict[str, Any] = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        }

        # Deterministic when temperature is 0 / very low; otherwise sample
        if temperature and temperature > 0.01:
            gen_kwargs.update({
                "do_sample": True,
                "temperature": float(temperature),
                "top_p": float(top_p),
            })
        else:
            gen_kwargs["do_sample"] = False

        output_ids = self.model.generate(**gen_kwargs)
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        result = {
            "response": response,
            "tokens_generated": len(new_tokens),
            "input_tokens": int(inputs["input_ids"].shape[1]),
            "finish_reason": "stop" if len(new_tokens) < max_new_tokens else "length",
        }
        logger.debug(
            "response_generated",
            response_length=len(response),
            tokens=len(new_tokens),
            in_tokens=result["input_tokens"],
        )
        return result

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded
