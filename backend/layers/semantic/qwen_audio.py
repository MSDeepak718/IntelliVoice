"""
IntelliVoice — Qwen-Audio Semantic Understanding

Layer 3: Converts speech embeddings into semantic understanding.
Supports multilingual audio understanding including Indian languages
and code-mixed conversations.
"""

from __future__ import annotations

import tempfile
from typing import Optional

import torch
import torchaudio

from config.logging_config import get_logger
from config.model_registry import ModelRegistry

logger = get_logger("qwen_audio")


class QwenAudioUnderstanding:
    """
    Qwen-Audio-based audio-semantic understanding.

    Responsibilities:
        - Speech understanding (what was said + how it was said)
        - Audio event understanding
        - Multilingual reasoning (EN, HI, TA, TE, code-mixed)

    Architecture:
        Speech Embeddings → Audio Understanding → Semantic Embeddings
    """

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self._device: torch.device = torch.device("cpu")
        self._is_loaded = False
        self._config = ModelRegistry.QWEN_AUDIO

    async def load(self, device: torch.device = torch.device("cuda")) -> None:
        """Load Qwen-Audio with INT4 quantization."""
        if self._is_loaded:
            return

        logger.info("loading_qwen_audio", model=self._config.hf_model_id)

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

            # Configure INT4 quantization
            quantization_config = None
            if self._config.quantization_config and device.type == "cuda":
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

            logger.info("qwen_audio_loaded", device=str(device))
        except Exception as e:
            logger.error("qwen_audio_load_failed", error=str(e))
            raise

    @torch.inference_mode()
    def understand_audio(
        self,
        waveform: torch.Tensor,
        sample_rate: int = 16000,
        prompt: str = "Describe what the speaker is saying, including their emotion and intent.",
        max_new_tokens: int = 256,
    ) -> dict:
        """
        Understand audio content using Qwen-Audio.

        Args:
            waveform: Audio tensor [1, T] or [T].
            sample_rate: Audio sample rate.
            prompt: Instruction prompt for the model.
            max_new_tokens: Maximum tokens to generate.

        Returns:
            Dict with:
                - 'transcription': What was said
                - 'understanding': Semantic understanding
                - 'raw_output': Full model output
        """
        if not self._is_loaded:
            raise RuntimeError("Qwen-Audio not loaded.")

        # Save waveform to temp file (Qwen-Audio reads audio files)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            torchaudio.save(tmp.name, waveform.cpu().float(), sample_rate)
            audio_path = tmp.name

        try:
            # Build the Qwen-Audio query
            query = self.tokenizer.from_list_format([
                {"audio": audio_path},
                {"text": prompt},
            ])

            response, _ = self.model.chat(
                self.tokenizer,
                query=query,
                history=None,
                max_new_tokens=max_new_tokens,
            )

            result = {
                "understanding": response,
                "raw_output": response,
            }

            logger.debug(
                "audio_understood",
                response_length=len(response),
            )

            return result
        finally:
            import os
            os.unlink(audio_path)

    @torch.inference_mode()
    def transcribe(
        self,
        waveform: torch.Tensor,
        sample_rate: int = 16000,
        language: str = "auto",
    ) -> str:
        """
        Transcribe audio to text.

        Args:
            waveform: Audio tensor.
            sample_rate: Audio sample rate.
            language: Target language or 'auto' for detection.

        Returns:
            Transcription string.
        """
        prompt = "Transcribe the speech in this audio exactly as spoken."
        if language != "auto":
            prompt = f"Transcribe the speech in this audio in {language}."

        result = self.understand_audio(
            waveform=waveform,
            sample_rate=sample_rate,
            prompt=prompt,
        )
        return result["understanding"]

    @torch.inference_mode()
    def analyze_intent(
        self,
        waveform: torch.Tensor,
        sample_rate: int = 16000,
    ) -> dict:
        """
        Analyze the speaker's intent from audio.

        Returns:
            Dict with intent, emotion, language, and summary.
        """
        prompt = (
            "Analyze this audio and respond in JSON format with these fields:\n"
            '- "transcription": exact words spoken\n'
            '- "language": detected language\n'
            '- "intent": speaker\'s intent (question/request/statement/greeting/complaint)\n'
            '- "emotion": detected emotion (happy/sad/angry/neutral/frustrated/excited)\n'
            '- "summary": brief summary of what the speaker wants'
        )

        result = self.understand_audio(
            waveform=waveform,
            sample_rate=sample_rate,
            prompt=prompt,
            max_new_tokens=512,
        )

        # Try to parse as JSON, fallback to raw
        import json
        try:
            parsed = json.loads(result["understanding"])
            return parsed
        except json.JSONDecodeError:
            return {
                "transcription": result["understanding"],
                "language": "unknown",
                "intent": "unknown",
                "emotion": "unknown",
                "summary": result["understanding"],
            }

    def offload_to_cpu(self) -> None:
        """Offload model to CPU."""
        if self.model is not None:
            self.model = self.model.cpu()
            torch.cuda.empty_cache()
            self._device = torch.device("cpu")
            logger.info("qwen_audio_offloaded_to_cpu")

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded
