"""
IntelliVoice — Fast LLM Reasoning Layer

Core reasoning engine.

Model: Qwen/Qwen2.5-7B-Instruct
    - Fast reasoning model
    - Strong multilingual + code reasoning
    - Fits comfortably in VRAM with 4-bit quantization

Responsibilities:
    - Intent understanding
    - Planning and problem solving
    - Question answering
    - Dialogue management
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import threading
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)

from config import get_settings
from config.logging_config import get_logger
from config.model_registry import ModelRegistry

logger = get_logger("qwen_reasoning")


SYSTEM_PROMPT = """\
You are Dhurva, a multilingual AI voice assistant. \
You communicate naturally and empathetically.

You will referred as Dhurva and you should also introduce as like that.

Key behaviors:
- Respond in the same language the user speaks (English, Hindi, Tamil, Telugu, or code-mixed)
- Match the user's emotional tone appropriately
- Keep responses concise and conversational (suitable for speech output, ideally 1-3 sentences)
- Be helpful, warm, and professional
- If the user sounds distressed, be empathetic first before solving their problem
- Avoid long lists, markdown, emojis, or code blocks — your response will be spoken aloud
- Do NOT use emojis, asterisks, hashtags, special formatting, or any other symbols that cannot be naturally spoken.
- If the user's question is ambiguous, ask one short clarifying question
- Do NOT use <think> tags or output internal reasoning. Provide the final spoken response directly.

You receive context about the user's:
- Speech transcription
- Recent conversation turns

Use this context to generate appropriate spoken responses."""


class InterruptibleStoppingCriteria(StoppingCriteria):
    def __init__(self):
        self.stop_now = False

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return self.stop_now


class FastReasoner:
    """
    Fast LLM reasoning engine.
    """

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self._device: torch.device = torch.device("cpu")
        self._is_loaded = False
        self._config = ModelRegistry.FAST_LLM
        self._settings = get_settings()
        self._generation_lock = threading.Lock()
        self._stopping_criteria = InterruptibleStoppingCriteria()

    def cancel_generation(self):
        """Abort any ongoing LLM generation."""
        self._stopping_criteria.stop_now = True

    async def load(self, device: torch.device = torch.device("cuda")) -> None:
        """Load Fast LLM model."""
        if self._is_loaded:
            return
        logger.info("loading_fast_llm", model=self._config.hf_model_id)

        try:
            quantization_config = None
            if device.type == "cuda" and self._config.quantization_config:
                quantization_config = BitsAndBytesConfig(
                    **{
                        "load_in_4bit": self._config.quantization_config.get("load_in_4bit", True),
                        "bnb_4bit_compute_dtype": torch.float16,
                        "bnb_4bit_quant_type": self._config.quantization_config.get(
                            "bnb_4bit_quant_type", "nf4"
                        ),
                        "bnb_4bit_use_double_quant": self._config.quantization_config.get(
                            "bnb_4bit_use_double_quant", True
                        ),
                    }
                )

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
                dtype=torch.float16,
                attn_implementation="sdpa",  # scaled-dot-product attention (no extra deps)
            )
            self.model.eval()
            self._device = device
            self._is_loaded = True

            num_params = sum(p.numel() for p in self.model.parameters()) / 1e9
            logger.info(
                "fast_llm_loaded",
                params_b=f"{num_params:.1f}",
                device=str(device),
                quant="nf4_double_quant",
            )
        except Exception as e:
            logger.error("fast_llm_load_failed", error=str(e))
            raise

    def _build_messages(
        self,
        user_message: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        system_prompt: str = SYSTEM_PROMPT,
    ) -> List[Dict[str, str]]:
        """Compose the chat-template messages list."""
        messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]

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
        system_prompt: str = SYSTEM_PROMPT,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Generate a response (non-streaming, used for text chat mode).

        Args:
            user_message: Transcribed user speech or typed text.
            conversation_history: Previous turns (most recent last).
            system_prompt: Override the default system prompt.
            max_new_tokens: Cap on new tokens.
            temperature: Sampling temperature.
            top_p: Nucleus sampling.

        Returns:
            Dict with 'response', 'tokens_generated', 'input_tokens', 'finish_reason'.
        """
        if not self._is_loaded:
            raise RuntimeError("Fast LLM not loaded.")

        # Request any running generation to stop immediately
        self.cancel_generation()

        with self._generation_lock:
            # Reset the stop flag for this new generation
            self._stopping_criteria.stop_now = False

            max_new_tokens = max_new_tokens or self._settings.max_new_tokens
            temperature = temperature if temperature is not None else self._settings.temperature
            top_p = top_p if top_p is not None else self._settings.top_p

            messages = self._build_messages(
                user_message=user_message,
                conversation_history=conversation_history,
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
                gen_kwargs.update(
                    {
                        "do_sample": True,
                        "temperature": float(temperature),
                        "top_p": float(top_p),
                    }
                )
            else:
                gen_kwargs["do_sample"] = False

            gen_kwargs["stopping_criteria"] = StoppingCriteriaList([self._stopping_criteria])

            output_ids = self.model.generate(**gen_kwargs)
            new_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
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

    async def stream_generate_response(
        self,
        user_message: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        system_prompt: str = SYSTEM_PROMPT,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ):
        """Streaming version of generate_response — yields tokens as they are generated."""
        import asyncio
        if not self._is_loaded:
            raise RuntimeError("Fast LLM not loaded.")

        self.cancel_generation()

        # Reset flag for this new generation
        self._stopping_criteria.stop_now = False

        max_new_tokens = max_new_tokens or self._settings.max_new_tokens
        temperature = temperature if temperature is not None else self._settings.temperature
        top_p = top_p if top_p is not None else self._settings.top_p

        messages = self._build_messages(
            user_message=user_message,
            conversation_history=conversation_history,
            system_prompt=system_prompt,
        )

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self._device)

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        gen_kwargs: Dict[str, Any] = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            "streamer": streamer,
            "stopping_criteria": StoppingCriteriaList([self._stopping_criteria]),
        }

        if temperature and temperature > 0.01:
            gen_kwargs.update(
                {
                    "do_sample": True,
                    "temperature": float(temperature),
                    "top_p": float(top_p),
                }
            )
        else:
            gen_kwargs["do_sample"] = False

        def generate_with_stream(**kwargs):
            if self._device.type == "cuda":
                stream = torch.cuda.Stream()
                with torch.cuda.stream(stream):
                    self.model.generate(**kwargs)
            else:
                self.model.generate(**kwargs)

        thread = threading.Thread(target=generate_with_stream, kwargs=gen_kwargs)
        thread.start()

        for new_text in streamer:
            yield new_text
            await asyncio.sleep(0)  # Yield control to event loop

        thread.join()

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded
