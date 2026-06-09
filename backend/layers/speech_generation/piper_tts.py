"""
IntelliVoice — Piper TTS Speech Generation

Layer 10: Converts semantic response text into natural speech.
Uses Piper TTS for extremely fast, stable, fully-local generation.
"""

from __future__ import annotations

import os
import io
import wave
from typing import AsyncGenerator, Optional, Tuple

import torch
import numpy as np

from config.logging_config import get_logger
from config.model_registry import ModelRegistry
from config.settings import get_settings

logger = get_logger("piper_tts")


class PiperSynthesizer:
    """
    Piper TTS text-to-speech synthesizer.
    
    Capabilities:
        - Extremely fast, stable CPU/GPU synthesis
        - Runs entirely offline without complex dependency conflicts
    """

    def __init__(self):
        self.model = None
        self._is_loaded = False
        self._config = ModelRegistry.PIPER
        self._sample_rate = 22050  # Usually 22050 for Piper
        self.settings = get_settings()

    async def load(self, device: torch.device = None) -> None:
        """Load Piper TTS model."""
        if self._is_loaded:
            return

        logger.info("loading_piper_tts", model=self._config.hf_model_id)

        try:
            from piper import PiperVoice
            from huggingface_hub import hf_hub_download
            
            # Default to an excellent medium-quality English voice
            repo_id = "rhasspy/piper-voices"
            model_file = "en/en_US/amy/medium/en_US-amy-medium.onnx"
            config_file = "en/en_US/amy/medium/en_US-amy-medium.onnx.json"
            
            model_path = self.settings.piper_model_path
            
            if not model_path or not os.path.exists(model_path):
                logger.info("downloading_piper_voice", voice="en_US-amy-medium")
                model_path = hf_hub_download(repo_id=repo_id, filename=model_file)
                config_path = hf_hub_download(repo_id=repo_id, filename=config_file)
            else:
                config_path = model_path + ".json"

            # Load Piper model
            # Piper automatically uses ONNX Runtime GPU if available
            # Note: Piper uses `use_cuda=True` flag
            use_cuda = (device and device.type == "cuda") or torch.cuda.is_available()
            self.model = PiperVoice.load(model_path, config_path=config_path, use_cuda=use_cuda)
            self._sample_rate = self.model.config.sample_rate
            
            self._is_loaded = True
            logger.info("piper_loaded", device="cuda" if use_cuda else "cpu")

        except Exception as e:
            logger.error("piper_load_failed", error=str(e))
            self._is_loaded = False

    @torch.inference_mode()
    def synthesize(
        self,
        text: str,
        language: str = "english",
        speaker_embedding: Optional[torch.Tensor] = None,
        emotion: str = "neutral",
        speaking_rate: float = 1.0,
    ) -> Tuple[torch.Tensor, int]:
        """
        Synthesize speech from text using Piper.
        """
        if not self._is_loaded:
            raise RuntimeError("Piper TTS not loaded.")

        try:
            # Piper synthesizes directly to a Wave_write object
            wav_io = io.BytesIO()
            with wave.open(wav_io, "wb") as wav_file:
                # Set WAV parameters just in case, though Piper often overrides them
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(self._sample_rate)
                self.model.synthesize_wav(text, wav_file)
            
            # Read the generated WAV file back into a torch Tensor
            import torchaudio
            wav_io.seek(0)
            waveform, sr = torchaudio.load(wav_io)
            
            return waveform, sr

        except Exception as e:
            logger.error("synthesis_failed", error=str(e), text_length=len(text))
            raise

    def offload_to_cpu(self) -> None:
        """Piper manages its own ONNX Runtime session, no explicit offload needed."""
        pass

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def sample_rate(self) -> int:
        return self._sample_rate
