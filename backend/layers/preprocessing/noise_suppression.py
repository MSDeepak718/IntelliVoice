"""
IntelliVoice — DeepFilterNet Noise Suppression

Layer 1 component: Removes background noise from audio
for improved downstream model performance.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from config.logging_config import get_logger

logger = get_logger("noise_suppression")


class NoiseSuppressor:
    """
    Noise suppression using DeepFilterNet.

    Removes background noise, improves speech quality,
    and enhances noisy telephone audio for cleaner input
    to downstream encoders.
    """

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self._model = None
        self._df_state = None
        self._is_loaded = False

    async def load(self) -> None:
        """Load the DeepFilterNet model."""
        if self._is_loaded:
            return

        logger.info("loading_deepfilternet")
        try:
            # DeepFilterNet uses its own loading mechanism
            from df.enhance import enhance, init_df, load_audio, save_audio

            self._df_model, self._df_state, _ = init_df()
            self._enhance_fn = enhance
            self._is_loaded = True
            logger.info("deepfilternet_loaded")
        except ImportError:
            logger.warning(
                "deepfilternet_not_available",
                fallback="passthrough",
                hint="Install with: pip install deepfilternet",
            )
            self._is_loaded = False
        except Exception as e:
            logger.error("deepfilternet_load_failed", error=str(e))
            self._is_loaded = False

    def suppress_noise(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Apply noise suppression to a waveform.

        Args:
            waveform: Audio tensor [1, T] at 16kHz or the model's native rate.

        Returns:
            Denoised waveform tensor [1, T].
        """
        if not self._is_loaded:
            logger.debug("noise_suppression_skipped", reason="model_not_loaded")
            return waveform

        try:
            # DeepFilterNet expects numpy array in specific format
            if waveform.dim() > 1:
                audio_np = waveform.squeeze(0).cpu().numpy()
            else:
                audio_np = waveform.cpu().numpy()

            # Apply enhancement
            enhanced = self._enhance_fn(
                self._df_model,
                self._df_state,
                audio_np,
            )

            # Convert back to tensor
            enhanced_tensor = torch.from_numpy(enhanced).float()
            if enhanced_tensor.dim() == 1:
                enhanced_tensor = enhanced_tensor.unsqueeze(0)

            logger.debug(
                "noise_suppressed",
                input_rms=f"{torch.sqrt(torch.mean(waveform ** 2)).item():.4f}",
                output_rms=f"{torch.sqrt(torch.mean(enhanced_tensor ** 2)).item():.4f}",
            )

            return enhanced_tensor

        except Exception as e:
            logger.error("noise_suppression_failed", error=str(e))
            return waveform  # Passthrough on error

    def suppress_noise_numpy(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply noise suppression to a numpy array.

        Args:
            audio: Numpy array of audio samples.

        Returns:
            Denoised numpy array.
        """
        waveform = torch.from_numpy(audio).float()
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        result = self.suppress_noise(waveform)
        return result.squeeze(0).numpy()

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded
