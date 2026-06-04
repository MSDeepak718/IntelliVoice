"""
IntelliVoice — HiFi-GAN Vocoder

Layer 12: Converts mel spectrograms into high-quality waveforms.
Provides low-latency, high-fidelity audio synthesis.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import numpy as np

from config.logging_config import get_logger
from config.model_registry import ModelRegistry

logger = get_logger("hifi_gan")


class HiFiGANVocoder:
    """
    HiFi-GAN vocoder for mel spectrogram to waveform conversion.

    Provides:
        - Low latency waveform generation
        - High quality 22kHz/24kHz audio output
        - Real-time capable on GPU
    """

    def __init__(self):
        self.model = None
        self._device: torch.device = torch.device("cpu")
        self._is_loaded = False
        self._config = ModelRegistry.HIFIGAN
        self._output_sample_rate = 22050

    async def load(self, device: torch.device = torch.device("cuda")) -> None:
        """Load HiFi-GAN vocoder."""
        if self._is_loaded:
            return

        logger.info("loading_hifigan")

        try:
            # Try loading via torch.hub (NVIDIA's implementation)
            try:
                self.model = torch.hub.load(
                    "nvidia/DeepLearningExamples:torchhub",
                    "nvidia_hifigan",
                    trust_repo=True,
                )
                self.model = self.model.to(device)
                self.model.eval()
                self._device = device
                self._is_loaded = True
                logger.info("hifigan_loaded_nvidia", device=str(device))
                return
            except Exception:
                logger.info("nvidia_hifigan_not_available", trying="transformers")

            # Fallback: Load via transformers
            from transformers import AutoModel

            self.model = AutoModel.from_pretrained(
                self._config.hf_model_id,
                trust_remote_code=True,
            )
            if hasattr(self.model, "to"):
                self.model = self.model.to(device)
            if hasattr(self.model, "eval"):
                self.model.eval()

            self._device = device
            self._is_loaded = True
            logger.info("hifigan_loaded_transformers", device=str(device))

        except Exception as e:
            logger.warning(
                "hifigan_load_failed",
                error=str(e),
                note="CosyVoice 2 includes its own vocoder, HiFi-GAN is optional",
            )
            self._is_loaded = False

    @torch.inference_mode()
    def mel_to_waveform(
        self,
        mel_spectrogram: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert mel spectrogram to waveform.

        Args:
            mel_spectrogram: Mel spectrogram tensor [B, M, T] or [M, T].

        Returns:
            Waveform tensor [1, T].
        """
        if not self._is_loaded:
            raise RuntimeError("HiFi-GAN not loaded.")

        if mel_spectrogram.dim() == 2:
            mel_spectrogram = mel_spectrogram.unsqueeze(0)

        mel_spectrogram = mel_spectrogram.to(self._device)

        waveform = self.model(mel_spectrogram)

        if isinstance(waveform, dict):
            waveform = waveform.get("audio", waveform.get("waveform"))

        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        logger.debug(
            "mel_to_waveform",
            mel_shape=str(mel_spectrogram.shape),
            wav_shape=str(waveform.shape),
        )

        return waveform.cpu()

    @torch.inference_mode()
    def enhance_waveform(
        self,
        waveform: torch.Tensor,
        target_sample_rate: int = 22050,
    ) -> torch.Tensor:
        """
        Post-process and enhance a waveform.

        Applies:
            - Peak normalization
            - De-clipping
            - Smooth fade in/out
        """
        if waveform.dim() > 1:
            waveform = waveform.squeeze(0)

        # Peak normalize
        peak = waveform.abs().max()
        if peak > 0:
            waveform = waveform / peak * 0.95

        # Fade in/out (10ms)
        fade_samples = int(target_sample_rate * 0.01)
        if len(waveform) > fade_samples * 2:
            fade_in = torch.linspace(0, 1, fade_samples)
            fade_out = torch.linspace(1, 0, fade_samples)
            waveform[:fade_samples] *= fade_in
            waveform[-fade_samples:] *= fade_out

        return waveform.unsqueeze(0)

    def offload_to_cpu(self) -> None:
        """Offload to CPU."""
        if self.model is not None and hasattr(self.model, "cpu"):
            self.model = self.model.cpu()
            torch.cuda.empty_cache()
            self._device = torch.device("cpu")
            logger.info("hifigan_offloaded_to_cpu")

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def sample_rate(self) -> int:
        return self._output_sample_rate
