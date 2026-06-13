"""
IntelliVoice — OmniVoice Generation Layer

Layer 10: Converts semantic response text into natural speech.
Uses k2-fsa/OmniVoice for exceptionally fast, high-quality zero-shot TTS.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple
import torch
import numpy as np

from config.logging_config import get_logger
from config.model_registry import ModelRegistry

logger = get_logger("omnivoice_tts")


class OmniVoiceSynthesizer:
    def __init__(self):
        self.model = None
        self._is_loaded = False
        self._config = ModelRegistry.OMNIVOICE
        self._sample_rate = 24000

    async def load(self, device: torch.device = None) -> None:
        """Load OmniVoice directly into the main process."""
        if self._is_loaded:
            return

        logger.info("loading_omnivoice", model_id=self._config.hf_model_id)
        try:
            from omnivoice import OmniVoice
            
            dev_str = str(device) if device else "auto"
            
            self.model = OmniVoice.from_pretrained(
                self._config.hf_model_id,
                device_map=dev_str,
                dtype=torch.float16 if self._config.precision.value == "fp16" else torch.float32
            )
            self._is_loaded = True
            logger.info("omnivoice_loaded_successfully")
        except ImportError:
            logger.error("omnivoice_not_installed", hint="Run: uv pip install omnivoice")
            raise
        except Exception as e:
            logger.error("omnivoice_load_failed", error=str(e))
            raise

    def synthesize(
        self,
        text: str,
        language: str = "en",
        speaker_embedding: Optional[torch.Tensor] = None,
        emotion: str = "neutral",
        speaking_rate: float = 1.0,
    ) -> Tuple[torch.Tensor, int]:
        """
        Synthesize speech using OmniVoice.
        """
        if not self._is_loaded:
            raise RuntimeError("OmniVoice is not loaded.")
        
        try:
            # For zero-shot, we MUST pass ref_audio to maintain a consistent voice.
            # Without it, the diffusion model generates a random voice every time.
            kwargs = {"text": text}
            
            # Robust absolute path resolution to IntelliVoice/assets/kiara_ref.wav
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
            ref_path = os.path.join(base_dir, "assets", "Thiru.wav")
            
            if os.path.exists(ref_path):
                kwargs["ref_audio"] = ref_path
                kwargs["ref_text"] = "My strengths are problem solving, quick learning, and the ability to convert business requirements into working technical solutions. Currently, I am looking for opportunities where I can contribute as a software engineer, work on large scale systems, and continue learning from experienced teams." # Optional, but helps OmniVoice align phonetics
                logger.debug("using_reference_audio", path=ref_path)
            else:
                logger.warning("reference_audio_missing", expected_path=ref_path, hint="Place dhurva_ref.wav here to lock the voice!")
            
            # OmniVoice returns a list of np.ndarray with shape (T,)
            audio_arrays = self.model.generate(**kwargs)
            
            if not audio_arrays or len(audio_arrays) == 0:
                raise ValueError("OmniVoice returned empty audio.")

            waveform = torch.tensor(audio_arrays[0], dtype=torch.float32).unsqueeze(0)
            return waveform, self._sample_rate

        except Exception as e:
            logger.error("omnivoice_synthesis_failed", error=str(e), text_length=len(text))
            raise

    def offload_to_cpu(self) -> None:
        if self.model is not None and hasattr(self.model, "to"):
            self.model.to("cpu")
            torch.cuda.empty_cache()

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def sample_rate(self) -> int:
        return self._sample_rate
